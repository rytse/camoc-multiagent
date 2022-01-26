from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.env import AECEnv
import numpy as np
from gym.utils import seeding
from typing import Optional, List, Dict
from itertools import combinations
from scipy.spatial import distance_matrix
from gym import spaces
from pettingzoo.utils.agent_selector import agent_selector


class RotatorCoverageWorld:
    def __init__(self, n_agents: np.int, n_entities: np.int, entity_sizes: np.ndarray):
        self.n_agents = n_agents
        self.n_landmarks = n_entities - n_agents
        self.n_entities = n_entities
        self.entity_sizes = entity_sizes

        # World parameters
        self.dt = 0.1
        self.damping = 1.5e-2
        self.maxspeed = 2

        self.dim_p = 2  # (x,y)
        self.contact_force = np.float32(5e2)
        self.contact_margin = np.float32(1e-4)

        # Agent controls, every agent has an X, Y positon
        self.positions: np.ndarray = np.zeros(shape=(n_entities, self.dim_p))
        self.velocities: np.ndarray = np.zeros(shape=(n_entities, self.dim_p))

        self.ctrl_thetas: np.ndarray = np.zeros(n_entities)
        self.ctrl_speeds: np.ndarray = np.zeros(n_entities)  # [:, None]

        # Agent can do an action, action space for each agent is (X, Y)
        self.agent_actions: np.ndarray[np.ndarray] = np.zeros(
            shape=(n_entities, self.dim_p)
        )

        # Only agents are movable
        self.movables = np.zeros(n_entities, dtype=bool)
        self.movables[:n_agents] = 1
        self.targets = self.movables[:]
        self.targets = np.ones(n_entities, dtype=bool)  # [:, None]
        self.targets[:n_agents] = 0

        self.entity_sizes: np.ndarray = entity_sizes

        self.sizemat = self.entity_sizes[..., None] + self.entity_sizes[None, ...]
        self.diag_indices = np.diag_indices(self.positions.shape[0])

    def step(self) -> None:
        # heading.shape = 2,6,2
        heading: np.ndarray = np.vstack(
            [np.cos(self.ctrl_thetas), np.sin(self.ctrl_thetas)]
        )
        heading = heading.T
        self.velocities = (
            self.maxspeed * (self.ctrl_speeds * self.movables)[:, None] * heading
        )

        # Detect collisons
        dist_matrix = distance_matrix(self.positions, self.positions)
        self.collisions: np.ndarray = dist_matrix < self.sizemat
        self.collisions[self.diag_indices] = False
        # self.collisions[self.low_triangular_indices_positions] = False
        # prolly a smarter way to check
        if np.any(self.collisions):
            # Calculate collison forces
            penetrations = (
                np.logaddexp(0, -(dist_matrix - self.sizemat) / self.contact_margin)
                * self.contact_margin
                * self.collisions
            )

            forces_s: np.float32 = self.contact_force * penetrations * self.collisions
            diffmat: np.ndarray = (
                self.positions[:, None, :] - self.positions
            )  # skew symetric

            forces_v = diffmat * forces_s[..., None]

            # 4 integrate collsion forces
            self.velocities -= np.sum(forces_v, axis=0) * self.dt

        # 5 integrate damping
        self.velocities -= self.velocities * self.damping

        # Integrate position
        self.positions += self.velocities * self.dt

    def dumb_collision_fix(self):
        positions_copy = self.positions[:]
        square_sizes = np.square(self.entity_sizes)
        for idx, pos in enumerate(self.positions):
            tmp = np.square(positions_copy - pos)
            indices = np.where(tmp < square_sizes)

    @property
    def landmarks(self) -> np.ndarray:
        return self.positions[self.n_agents :, :]

    @property
    def agents(self) -> np.ndarray:
        return self.positions[: self.n_agents, :]

    @property
    def agent_velocities(self) -> np.ndarray:
        return self.velocities[: self.n_agents, :]

    def observation(self, agent_index) -> np.ndarray:
        """
        WARNING: DOES NOT RETURN COMMUNICATION
        """
        # Calculate distances and velocities
        vec_to_targets = self.landmarks - self.agents[agent_index]  # targets
        dist_to_targets = np.linalg.norm(vec_to_targets, axis=1)
        vec_to_agents = self.agents - self.agents[agent_index]  # agents
        dist_to_agents = np.linalg.norm(vec_to_agents, axis=1)
        vel_to_agents = self.agent_velocities[agent_index] - self.agent_velocities
        speed_to_agents = np.linalg.norm(vel_to_agents, axis=1)

        # Calculate angles to landmarks
        angles_to_targets = np.arctan2(vec_to_targets[:, 1], vec_to_targets[:, 0])

        # Agents / landmarks have no intrinsic order, so we sort by distance
        targets_order = np.argsort(dist_to_targets)  # targets
        dist_to_targets = dist_to_targets[targets_order]
        angles_to_targets = angles_to_targets[targets_order]
        agents_order = np.argsort(dist_to_agents)  # agents
        dist_to_agents = dist_to_agents[agents_order]
        speed_to_agents = speed_to_agents[agents_order]

        obs = np.concatenate(
            [
                dist_to_targets,
                angles_to_targets,
                dist_to_agents[1:],
                speed_to_agents[1:],
            ]
        )

        # Don't forget that we included the agent itself in the list of others
        return obs

    def reset(self, np_random) -> None:
        """
        Resets the world
        """
        self.positions = np_random.uniform(-3, +3, size=(self.positions.shape))
        self.velocities[:] = 0
        self.agent_actions[:] = 0

        self.ctrl_thetas[:] = 0
        self.ctrl_speeds[:] = 0


# Could inherit base Scenario
class RotatorCoverageScenario:
    def make_world(self, N=3, AGENT_SIZE=0.15, LANDMARK_SIZE=1.0, N_THETA=10):
        entity_sizes: np.ndarray = np.array(
            [AGENT_SIZE for _ in range(N)] + [LANDMARK_SIZE]
        )
        print(entity_sizes)
        world = RotatorCoverageWorld(N, N + 1, entity_sizes)

        # Set world properties first
        # world.collaborative = True

        # add agents

        # add landmarks

        # Generate mesh that approximates the target
        theta = np.linspace(0, 2 * np.pi, N_THETA)
        xx = LANDMARK_SIZE * np.cos(theta)
        yy = LANDMARK_SIZE * np.sin(theta)
        self.target_mesh = np.rollaxis(np.array([xx, yy]), 1)

        return world

    def global_reward(self, world):
        """
        Calculate global reward for the world in its current state.
        """

        # Get distance penalty
        headings = np.array([np.cos(world.ctrl_thetas), np.sin(world.ctrl_thetas)]).T
        headpos = (
            world.positions + headings * (world.entity_sizes * ~world.targets)[:, None]
        )

        # dists = distance_matrix(headpos, headpos)
        dists = distance_matrix(world.positions, world.positions)
        # indices where (agent, landmark) = True
        # (agent, agent) (landmark, landmark) = False
        mask = ~np.logical_xor(~world.targets[:, None], world.targets[None, :])
        relevant_dists = dists * mask
        dist_penalty = np.square(np.sum(relevant_dists))

        # Get coverage reward
        diffmat = world.positions[:, None, :] - world.positions
        rel_thetas = np.arctan2(diffmat[:, :, 1], diffmat[:, :, 0])
        coverage_reward = np.var(rel_thetas)

        # Combine the two objectives
        return 10 * coverage_reward - dist_penalty

    def reset_world(self, world, np_random):
        world.reset(np_random)

    def observation(self, agent_index: int, world: RotatorCoverageWorld) -> np.ndarray:
        return world.observation(agent_index)


class RotatorCoverageEnv(AECEnv):
    def __init__(
        self,
        scenario: RotatorCoverageScenario,
        world: RotatorCoverageWorld,
        max_cycles: int,
        continuous_actions: bool = True,
        local_ratio: bool = None,
    ):
        super().__init__()
        self.seed()
        self.metadata = {"render.modes": ["human", "rgb_array"]}

        self.continuous_actions = True

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents: List[str] = [
            str(i) for i in range(self.world.n_agents)
        ]  # list(range(self.world.n_agents))
        self.possible_agents = self.agents[:]
        # Yes this is stupid... copying their old structure for now
        self._index_map = {agent: idx for idx, agent in enumerate(self.agents)}
        self._agent_selector = agent_selector(self.agents)

        self.n_agents = self.world.agents.shape[0]

        # Initialize action spaces
        self.action_spaces: Dict(int, spaces.Box) = dict()
        self.observation_spaces: Dict(int, spaces.Box) = dict()
        space_dim = self.world.dim_p * 2 + 1
        # We don't set a communication channel
        # iterate over agents
        state_dim = 0
        for i in range(len(self.world.agents)):
            obs_dim = len(self.scenario.observation(i, self.world))
            state_dim += obs_dim

            self.action_spaces[i] = spaces.Box(
                low=np.array([-np.pi, 0]), high=np.array([np.pi, 1]), dtype=np.float32
            )
            # Observations:
            obsmin = np.concatenate(
                [
                    [-np.inf] * self.world.n_landmarks,
                    [-np.pi] * self.world.n_landmarks,
                    [-np.inf] * (self.world.n_agents - 1),
                    [0] * (self.world.n_agents - 1),
                ]
            )

            obsmax = np.concatenate(
                [
                    [np.inf] * self.world.n_landmarks,
                    [np.pi] * self.world.n_landmarks,
                    [np.inf] * (self.world.n_agents - 1),
                    [self.world.maxspeed * 2] * (self.world.n_agents - 1),
                ]
            )

            self.observation_spaces[i] = spaces.Box(
                low=obsmin, high=obsmax, dtype=np.float32
            )

        # state space is the sum of all the local observation spaces it seems? shape wise taht is
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps: int = 0
        self.current_actions: List[Optional[spaces.Box]] = [None] * self.n_agents

        self.viewer = None

    def observation_space(self, agent_index: str) -> spaces.Box:
        return self.observation_spaces[int(agent_index)]

    def action_space(self, agent_index: str) -> spaces.Box:
        return self.action_spaces[int(agent_index)]

    def seed(self, seed: Optional[int] = None) -> None:
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent: str) -> np.ndarray:
        # When I rewrite supersuit I won't have to pass strings
        # Literally what the fuck
        return self.scenario.observation(int(agent), self.world)

    def state(self):
        states = tuple(
            self.scenario.observation(self.world.agents[agent], self.world).astype(
                np.float32
            )
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self):
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}

        self._cumulative_rewards = {name: 0.0 for name in self.agents}

        self.dones: Dict(str, bool) = {str(i): False for i in self.agents}
        # INFOS BECAUSE FUCKING KILL ME
        self.infos = {name: {} for name in self.agents}

        self._reset_render()

        # TODO: need to recall agent select reset
        self.steps = 0  # Not sure what this does
        self.agent_selection = self._agent_selector.reset()
        # TODO: current_actions??
        self.current_actions = [None] * self.n_agents

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            self.world.ctrl_thetas[i] = action[0]
            self.world.ctrl_speeds[i] = action[1]

        self.world.step()

        for a, agent in enumerate(self.agents):
            self.rewards[agent] = self.scenario.global_reward(self.world)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(
                action
            )  # TODO: Implement this reference in AECEnv

        curr_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.dones[str(a)] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[curr_agent] = 0
        self._accumulate_rewards()

    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

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
            for idx, (entity, entity_size) in enumerate(
                zip(self.world.positions, self.world.entity_sizes)
            ):
                geom = rendering.make_circle(entity_size)
                xform = rendering.Transform()
                if idx < self.world.n_agents:
                    geom.set_color(0.100, 0.100, 0.100, alpha=0.5)
                else:
                    geom.set_color(0.20, 0.6, 0.10)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            """
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            """

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            """
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1
            """

        """
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for idx, other in enumerate(self.world.agents):
            if other.silent:
                continue
            if np.all(other.c == 0):
                word = '_'
            elif self.continuous_actions:
                word = '[' + ",".join([f"{comm:.2f}" for comm in other.c]) + "]"
            else:
                word = alphabet[np.argmax(other.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)
        """

        # update bounds to center around agent
        all_poses = [pos for pos in self.world.positions]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        # print(self.world.positions.shape)
        for e, entity in enumerate(self.world.positions):
            self.render_geoms_xform[e].set_translation(*entity)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()
