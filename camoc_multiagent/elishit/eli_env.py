import torch

# from utils import Box
class Box:
    def __init__(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        dtype: torch.dtype,
        n_environments=1,
    ):
        self.low = low
        self.high = high
        self.dtype = dtype
        self.n_environments = n_environments
        assert (
            self.low.shape == self.high.shape
        ), "Box low and high shapes must be the same"

    @property
    def shape(self):
        return self.low.shape


import numpy as np  # Should eventually switch to not numpy?


class MultiEnvRotatorWorld:
    """
    TODO: Find some way of making pair distance faster?
    TODO: Write Cuda Layer if want ragged n_agents
    """

    def __init__(
        self,
        n_agents_per_environment: int,
        n_landmarks_per_environment: int,
        n_environments: int,
        use_gpu=True,
    ):
        # Set device first
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        # Set Passed Parameters
        self.n_agents_per_environment = n_agents_per_environment
        self.n_landmarks_per_environment = n_landmarks_per_environment
        self.n_entities_per_environment = (
            self.n_agents_per_environment + self.n_landmarks_per_environment
        )
        self.n_environments = n_environments

        # For now all agents are size .1, all landmarks are size .5
        self.entity_sizes = torch.zeros(
            (self.n_environments, self.n_entities_per_environment, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self.entity_sizes[
            :
        ] = 1.0  # set all to landmark and then overwrite agent positions
        self.entity_sizes[
            :, : self.n_agents_per_environment, :
        ] = 0.15  # set agent sizes

        # Generic world parameters
        self.dt: float = 0.5  # time step size
        self.damping: float = 1.5e-2
        self.max_speed: float = 2.0
        self.dim_p: int = 2  # agents move x,y
        # Some physics parameters, not sure if cpu or gpu
        self.contact_force: float = 1e2
        self.contact_margin: float = 1e-4

        # Entities have: position, velocity, ctrl parameters
        # TODO: Possibly don't give landmarks velocities or control parameters if possible
        self.positions = torch.zeros(
            (self.n_environments, self.n_entities_per_environment, self.dim_p),
            device=self.device,
            dtype=torch.float32,
        )
        self.velocities = torch.zeros(
            (self.n_environments, self.n_agents_per_environment, self.dim_p),
            device=self.device,
            dtype=torch.float32,
        )

        # Control parameters
        self.ctrl_thetas = torch.zeros(
            (self.n_environments, self.n_agents_per_environment, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self.ctrl_speeds = torch.zeros(
            (self.n_environments, self.n_agents_per_environment, 1),
            device=self.device,
            dtype=torch.float32,
        )

        # Agents specifically also have actions, TODO: stop allocating actiosn for entities
        # self.agent_actions = torch.zeros((self.n_environments, self.n_entities_per_environment, self.dim_p), device=self.device, dtype=torch.float32)
        self.movables = torch.zeros(
            (self.n_environments, self.n_entities_per_environment, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self.movables[:, : self.n_agents_per_environment] = 1  # only agents can move

        # Size matrix for each environment
        self.size_matrix = self.entity_sizes * self.entity_sizes.reshape(
            self.n_environments, 1, self.n_entities_per_environment
        )

        self.inv_eye = torch.logical_not(
            torch.eye(self.n_entities_per_environment).repeat(self.n_environments, 1, 1)
        )
        self.inv_eye = self.inv_eye.to(self.device)
        self.size_matrix = self.size_matrix * self.inv_eye
        # This is faster but I'll deal with this later
        # self.heading = torch.cat([torch.sin(self.ctrl_thetas), torch.cos(self.ctrl_thetas)], dim=2) # It is noticably better to update this
        # breakpoint()
        self.previous_angle_to_targets = torch.zeros(
            (self.n_environments, self.n_agents_per_environment, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self.previous_dist_to_targets = torch.zeros(
            (self.n_environments, self.n_agents_per_environment, 1),
            device=self.device,
            dtype=torch.float32,
        )

    def mem(self, msg=""):
        print(f"{msg}: {torch.cuda.memory_allocated(self.device)} ")

    def step(self) -> None:
        # Step 1: Update velocities with respect to ctrl_theta and ctrl_speed (modified by RL actions)
        heading: torch.Tensor = torch.cat(
            [torch.cos(self.ctrl_thetas), torch.sin(self.ctrl_thetas)], dim=2
        )
        # breakpoint()
        self.velocities = self.max_speed * self.ctrl_speeds * heading
        # breakpoint()
        # if torch.isnan(self.velocities).any():
        #    breakpoint()

        """
         # Step 2: Check collisions
        dist_matrix = torch.cdist(self.positions.clone(), self.positions.clone(), p=2)

        collisions = dist_matrix < self.size_matrix
        collisions *= self.inv_eye

        # Step 3: calculate collision forces
        penetrations: torch.Tensor = torch.log(
                        1.00001000005 + (
                        torch.exp(  -(dist_matrix - self.size_matrix + 1e-4)/ self.contact_margin)
                        ) 
        )* self.contact_margin * collisions
        forces_s: torch.Tensor = self.contact_force * (penetrations * collisions)[:, :self.n_agents_per_environment, :self.n_agents_per_environment]
        diff_matrix: torch.Tensor = self.positions[:, :self.n_agents_per_environment, None, :] - self.positions[:, None, :self.n_agents_per_environment, :] # broadcasting is so cool
        forces_v = diff_matrix * forces_s[..., None]

        # Step 4: Integrate collision forces
        #breakpoint()
        self.velocities += forces_v.sum(dim=1) * self.dt #[:, :self.n_agents_per_environment] * self.dt
        
        """

        # self.velocities[:, self.n_agents_per_environment:, :] = 0 # landmarks really shouldn't be getting actions

        # Step 5: Handle damping... not going to damp for now
        self.velocities -= self.velocities * (1 - self.damping)
        self.velocities = torch.clip(self.velocities, -self.max_speed, self.max_speed)
        # if torch.isnan(self.velocities).any():
        # breakpoint()
        # Step 6: Integrate position
        self.positions[:, : self.n_agents_per_environment] += self.velocities * self.dt

    @property
    def landmarks(self) -> torch.Tensor:
        # (n_envs, n_landmarks_per_env, 2)
        return self.positions[:, self.n_agents_per_environment :, :]

    @property
    def agents(self) -> torch.Tensor:
        # (n_envs, n_agents_per_env, 2)
        return self.positions[:, : self.n_agents_per_environment, :]

    @property
    def agent_velocities(self) -> torch.Tensor:
        return self.velocities[:, : self.n_agents_per_environment, :]

    def observation(self):
        # (n_envs, n_agents, n_landmarks, 2)
        vec_agents_to_targets = (
            self.landmarks[:, None, :, :] - self.agents[:, :, None, :]
        )
        dist_agents_to_targets = torch.norm(
            vec_agents_to_targets, dim=3
        )  # (n_envs, n_agents, n_landmarks)
        # This should be checked just to make sure
        angles_to_targets = torch.atan2(
            vec_agents_to_targets[:, :, :, 1], vec_agents_to_targets[:, :, :, 0]
        )

        vec_agents_to_agents = (
            self.agents[:, None, :, :] - self.agents[:, :, None, :]
        )  # yes includes self
        dist_agents_to_agents = torch.norm(vec_agents_to_agents, dim=3)

        vel_agents_to_agents = (
            self.agent_velocities[:, None, :, :] - self.agent_velocities[:, :, None, :]
        )
        speed_agents_to_agents = torch.norm(vel_agents_to_agents, dim=3)
        # (n_envs, n_agents, n_agents[:-landmarks])
        sorted_speeds, _ = torch.sort(speed_agents_to_agents, dim=2)
        sorted_speeds = sorted_speeds[:, :, 1:]

        sorted_dists, _ = torch.sort(dist_agents_to_agents, dim=2)
        sorted_dists = sorted_dists[:, :, 1:]
        # breakpoint()
        # Ryan ordered here but this seems, weird to me, the order should stay the order...
        # breakpoint()

        # clipping b/c me and ryan are stupid little fucking idiots
        sorted_speeds = torch.clip(sorted_speeds, 0, 2 * self.max_speed)
        # if dist_agents_to_agents.max() > 10:
        #    breakpoint()

        # (n_envs, n_agents, 12)
        obs = torch.cat(
            [
                dist_agents_to_targets,
                angles_to_targets,
                sorted_dists,
                sorted_speeds,
                self.previous_dist_to_targets,
                self.previous_angle_to_targets,
            ],
            dim=2,
        )
        # breakpoint()
        self.previous_dist_to_targets[:] = dist_agents_to_targets.clone()
        self.previous_angle_to_targets[:] = angles_to_targets.clone()
        return obs

    def reset(self):
        """
        Resets ALL environments
        """
        self.positions = (
            torch.rand(self.positions.shape, device=self.device) - 0.5
        ) * 6  # uh idk -3, 3 or something who knows
        self.velocities[:] = 0
        # self.agent_actions[:] = 0

        self.ctrl_speeds[:] = 0
        self.ctrl_thetas[:] = 0

    def global_reward(self):
        headings: torch.Tensor = torch.cat([self.ctrl_thetas, self.ctrl_speeds], dim=2)
        headpos = self.positions.clone()[:]

        headpos[:, : self.n_agents_per_environment] += (
            headings * self.entity_sizes[:, : self.n_agents_per_environment]
        )

        dists = torch.cdist(headpos, headpos)
        # mask away landmarks from reward
        mask = torch.logical_xor(self.movables, torch.logical_not(self.movables))
        relevant_dists = dists * mask
        # [:, None] to keep shape (n_environments, 1) rather than (n_environments)
        dist_penalty = relevant_dists.sum(dim=[1, 2]).square()[:, None]

        diff_matrix = self.positions[:, None, :, :] - self.positions[:, :, None, :]
        rel_thetas = torch.atan2(diff_matrix[:, :, :, 1], diff_matrix[:, :, :, 0])
        coverage_reward = torch.var(rel_thetas)

        return -dist_penalty / (coverage_reward + 1e6)


class MultiRotatorEnvironment:
    def __init__(self, n_agents: int, n_landmarks: int, n_worlds: int):
        self.device = torch.device("cuda")
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.n_worlds = n_worlds
        self.world = MultiEnvRotatorWorld(
            self.n_agents, self.n_landmarks, self.n_worlds
        )
        self.max_cycles = 25  # Just a random default all our agents "finish" at the same b/c we're not retarded
        self.current_step_mod_max_cycles = 0  # we can use this to control when done... swarmcover updated all dones every max_cycles steps

        self.action_dim = 2
        self.obs_dim = 7  # apparently 7 lolol
        self.viewer = None
        self.render_geoms = None
        self._setup_observation_and_action_space()

    def _setup_observation_and_action_space(self):
        obs_min = torch.cat(
            [
                torch.tensor(
                    [-np.inf] * self.n_landmarks,
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [-np.pi] * self.n_landmarks, device=self.device, dtype=torch.float32
                ),
                torch.tensor(
                    [-np.inf] * (self.n_agents), device=self.device, dtype=torch.float32
                ),
                torch.tensor(
                    [0] * (self.n_agents), device=self.device, dtype=torch.float32
                ),
            ]
        )

        obs_max = torch.cat(
            [
                torch.tensor(
                    [np.inf] * self.n_landmarks, device=self.device, dtype=torch.float32
                ),
                torch.tensor(
                    [np.pi] * self.n_landmarks, device=self.device, dtype=torch.float32
                ),
                torch.tensor(
                    [np.inf] * (self.n_agents), device=self.device, dtype=torch.float32
                ),
                torch.tensor(
                    [self.world.max_speed * 2] * (self.n_agents),
                    device=self.device,
                    dtype=torch.float32,
                ),
            ]
        )

        self.observation_space = Box(
            low=obs_min, high=obs_max, dtype=torch.float32, n_environments=self.n_worlds
        )

        # Action space
        self.action_space = Box(
            low=torch.tensor([-torch.pi, 0], device=self.device, dtype=torch.float32),
            high=torch.tensor([torch.pi, 1], device=self.device, dtype=torch.float32),
            dtype=np.float32,
            n_environments=self.n_worlds,
        )

    """
    just returns observations and rewards after stepping the world
    """

    def step(self, actions: torch.Tensor):
        # I'm not copying atm... this might cause really bad bugs we will see
        # breakpoint()
        # print(actions.shape)
        # print(f"ctrl_thetas {self.world.ctrl_thetas.shape}")
        # breakpoint()
        self.world.ctrl_thetas = actions[:, :, 0][
            :, :, None
        ]  # , -torch.pi, torch.pi) # the None is to keep the dimension nice
        self.world.ctrl_speeds = actions[:, :, 1][:, :, None]  # )#, 0, 1)
        # breakpoint()
        # breakpoint()
        self.world.step()
        self.current_step_mod_max_cycles += 1
        self.current_step_mod_max_cycles %= self.max_cycles
        return (
            self.world.observation(),
            0,
            0,
        )  # self.world.global_reward(), self.current_step_mod_max_cycles == 0

    def reset(self):
        self.world.reset()
        return self.world.observation()

    """
    if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)
    def render(self):
        from pettingzoo.mpe._mpe_utils import rendering
        background_color = (255,255,255)
        height = 300
        width = 300
        if self.has_setup_renderer == False:
            self.screen = pygame.display.set_mode((height*2,width*2))
            self.has_setup_renderer = True
            

        self.screen.fill(background_color)
        
        positions = self.world.positions[0] - self.world.positions.min()
        positions[:, 0] *= height/positions[:, 0].max()
        positions[:, 1] *= width/positions[:, 1].max()
        positions += width/2
        for (x,y) in positions[:-self.n_landmarks]:
            pygame.draw.circle(self.screen, (0,0,0), (int(x),int(y)), 5)
        
        (x,y) = positions[-1] #landmark

        pygame.draw.circle(self.screen, (0,255,0), (int(x),int(y)), 10)

        pygame.display.flip()
        breakpoint()
    """

    def render(self, mode="human"):
        from pettingzoo.mpe._mpe_utils import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for i, entity in enumerate(self.world.positions[0]):
                geom = rendering.make_circle(0.15)
                if i >= self.n_agents:
                    geom = rendering.make_circle(1.0)
                xform = rendering.Transform()
                if i < self.n_agents:
                    geom.set_color(*(0, 0, 0), alpha=0.5)
                else:
                    geom.set_color(*(0, 255, 0))
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

        # update bounds to center around agent
        all_poses = [pos for pos in self.world.positions[0].clone().cpu().numpy()]
        # breakpoint()
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        print(cam_range)
        if np.isnan(cam_range):
            breakpoint()
        # breakpoint()
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.positions[0]):
            self.render_geoms_xform[e].set_translation(*(entity.clone().cpu().numpy()))
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()
