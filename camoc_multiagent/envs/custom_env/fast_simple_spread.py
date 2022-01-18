from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.env import AECEnv
import numpy as np
from gym.utils import seeding
from typing import Optional, int, List
from itertools import combinations
from scipy.spatial import distance_matrix





# Could inherit base Scenario
class FastScenario:
    def make_world(self, N=3, AGENT_SIZE=0.15, LANDMARK_SIZE=1.5, N_THETA=10):
        world = FastWorld(N, 1, AGENT_SIZE, LANDMARK_SIZE)
        return world

    def global_reward(self, world):
        '''
        Calculate global reward for the world in its current state.
        '''

        # Get distance penalty
        headings = np.array([np.cos(world.ctrl_thetas), np.sin(world.ctrl_thetas)]) 
        headpos = world.positions + headings * world.entity_sizes * world.movables
        dists = distance_matrix(headpos, headpos)
        mask = world.movables[:, None] * world.movables[None, :]
        relevant_dists = dists * mask
        dist_penalty = np.sum(relevant_dists[:])

        # Get coverage reward
        diffmat = world.positions[:, None, :] - world.positions
        rel_thetas = np.arctan2(diffmat[:, :, 1], diffmat[:, :, 0])
        coverage_reward = np.var(rel_thetas)

        # Combine the two objectives
        return -dist_penalty / (coverage_reward + 1e-6)
    

    def reset_world(self, world, np_random):
        world.reset(np_random)


    def observation(self, agent_index: int, world) -> np.ndarray:
        return world.observation(self, agent_index)
        





class FastWorld:
    def __init__(self, n_agents: np.int, n_entities: np.int, entity_sizes: np.ndarray):
        self.n_agents = n_agents
        self.n_entities = n_entities
        self.entity_sizes = entity_sizes

        # World parameters
        self.dt = 0.1
        self.damping = 0.25

        self.dim_p = 2 # (x,y)
        self.contact_force = np.float32(1e2)
        self.contact_margin = np.float32(1e-3)


        # Agent controls, every agent has an X, Y positon
        self.positions: np.ndarray = np.zeros(shape=(n_entities, self.dim_p))
        self.velocities: np.ndarray = np.zeros(shape=(n_entities, self.dim_p))

        self.ctrl_thetas: np.ndarray = np.zeros(n_entities)
        self.ctrl_speeds: np.ndarray = np.zeros(n_entities)

        # Agent can do an action, action space for each agent is (X, Y)
        self.agent_actions: np.ndarray[np.ndarray] = np.zeros(shape=(n_entities, self.dim_p))

        # Only agents are movable
        self.movables = np.zeros(n_entities)
        self.movables[:n_agents] = 1

        self.entity_sizes: np.ndarray = entity_sizes

        self.sizemat = self.entity_sizes[..., None] * self.entity_sizes[None, ...]
        

        

    def step(self) -> None:
        angle: np.ndarray = np.array([np.cos(self.ctrl_thetas), np.sin(self.ctrl_thetas)])
        self.velocities = self.ctrl_speeds * angle * self.movables

        # 2 detect collisons
        dist_matrix = distance_matrix(self.positions, self.positions)
        collisions: np.ndarray = dist_matrix < self.sizematrix
        
        # prolly a smarter way to check
        if np.any(collisions):
            # 3 calculate collison forces  
            penetrations = np.logaddexp(0, -(dist_matrix - self.sizemat) / self.contact_margin) \
                                * self.contact_margin * collisions
            
            forces_s: np.float32 = self.contact_force * penetrations * collisions
            diffmat: np.ndarray = self.positions[:, None, :] - self.positions  # skew symetric
            
            forces_v = diffmat * forces_s[..., None]


            # 4 integrate collsion forces
            self.velocities += np.sum(forces_v, ax=0) * self.dt

        # 5 integrate damping
        self.velocities -= self.velocities * (1 - self.damping)

        # Integrate position
        self.positions += self.velocities * self.dt

  

    @property
    def landmarks(self) -> np.ndarray:
        return self.positions[self.n_agents+1:]

    @property
    def agents(self) -> np.ndarray:
        return self.positions[:self.n_agents]

    @property
    def agent_velocities(self) -> np.ndarray:
        return self.velocities[:self.n_agents]

    def observation(self, agent_index) -> np.ndarray:
        """
        WARNING: DOES NOT RETURN COMMUNICATION
        """
        
        # Agents / landmarks have no intrinsic order, so we sort by distance
        dist_to_targets = np.sort(np.linalg.norm(self.agents[agent_index] - self.landmarks, axis=1))
        dist_to_agents = np.sort(np.linalg.norm(self.agents[agent_index] - self.agents, axis=1))
        speed_to_agents = np.sort(np.linalg.norm(self.agent_velocities[agent_index] - self.agent_velocities, axis=1))

        # Don't forget that we included the agent itself in the list of others
        return np.concatenate([dist_to_targets[1:], dist_to_agents[1:], speed_to_agents[1:]])


    def reset(self, np_random) -> None:
        """
        Resets the world
        """
        self.positions =  np_random.unform(-1, +1, shape=(self.positions.shape))
        self.velocities[:] = 0
        self.agent_actions[:] = 0
        



class FastSimpleEnv:
    def __init__(self,
            scenario: FastScenario,
            world: FastWorld,
            max_cycles: int, 
            continuous_actions: bool = False, 
            local_ratio: bool = None):
        
        self.seed()
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)
        
        self.action_spaces = None 
        self.observation_spaces = None

        self.possible_agents = self.world.agents.copy()
        self.n_agents = self.world.agents.shape[0]

        # TODO: Initialize action spaces

    def observation_space(self, agent: int):
        raise Exception("Observation space funciton not implemented")


    def action_spaces(self, agent: int):
        raise Exception("Action space funciton not implemented not implemented")
    
    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)

    
    def observe(self, agent: int):
        return self.scenario.observation(agent, self.world)
    
    def state(self):
        raise Exception("State not implented yet...")
    
    def reset(self):
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = np.zeros(shape=self.agents.shape)
        self._cumulative_rewards = np.zeros(shape=self.agents.shape)
        
        self.dones: np.ndarray[bool] = np.zeros(shape=self.agents.shape)
        # I am omitting infos

        self._reset_render()

        # TODO: need to recall agent select reset
        self.steps = 0 #Not sure what this does

        # TODO: current_actions??
        self.current_actions = [None] * self.n_agents

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            # We don't need to ask if the agent is movable
            mdim = self.world.dim_p * 2 + 1
            # We don't do continuous actions

            # Literally wtf is this
            scenario_action.append(action % mdim)
            action //= mdim

            #if not agent.silent... we don't store that parameter
            scenario_action.append(action)
            # Need to fix this too
            self._set_action(scenario_action, i, self.action_spaces[i])
        
        self.world.step()

        self.rewards[:] = self.scenario.global_reward(self.world)
        

    def _set_action(self, action: List[int], agent_index: int, action_space, time=None)->None:
        # All our agents are movable
        # All our actions are discrete currently
        if action[0] == 1:
            self.world.agent_actions[agent_index][0] = 1.0
        if action[0] == 2:
            self.world.agent_actions[agent_index][0] = -1.0
        if action[0] == 3:
            self.world.agent_actions[agent_index][1] = -1.0
        if action[0] == 4:
            self.world.agent_actions[agent_index][1] = -1.0

        # We do not currently have a concept of acceleration
        sensitivity = 5.0
        self.world.agent_actions[agent_index] *= sensitivity
        action = action[1:]

        # Our agents are never silent or continuous, and do not communicate

        # To be honest I have no clue why we are passing what we are 
        assert len(action) == 0


    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step() # TODO: Implement this reference in AECEnv
        
        current_agent_idx = self.agent_selection
        self.agent_selection = (current_agent_idx + 1) % self.n_agents
        if self.agent_selection == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                self.dones[:] = True
        else:
            self._clear_rewards()
        
        #self._cumulative_rewards[cur_agent_idx] = 0
        self._accumulate_rewards()



    def _accumulate_rewards(self):
        raise Exception(")_accumulative rewards not yet implemented")

    def _clear_rewards(self):
        raise Exception("_clear_rewards is not eyt implemented")


    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
    def render(self, mode='human'):
        from . import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
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

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1

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

        # update bounds to center around agent
        all_poses = [entity.pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()
























class raw_env(FastSimpleEnv):
    def __init__(self, N=5, local_ratio=0.25, max_cycles=25, continuous_actions=True):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = FastScenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata['name'] = "swarm_cover_v1"





'''
class FastWorld:
    def __init__(self, n_agents, n_entities, AGENT_SIZE, LANDMARK_SIZE):
        #self.agents_x: np.ndarray = np.zeros(n_agents)
        #self.agents_y: np.ndarray = np.zeros(n_agents)
        self.agent_pos: np.ndarray = np.zeros(shape=(n_agents, 2))
        self.agent_color: np.ndarray = np.array([0.35, 0.35, 0.35]) # all agents same color
        self.agent_size = AGENT_SIZE

        self.entity_positions
        self.agent_actions = np.zeros(shape=(n_agents, 2))

        self.landmark_pos: np.ndarray = np.zeros(2)
        self.landmark_color: np.ndarray = np.array([0.25, 0.25, 0.25])

        self.dt: np.float32 = np.float32(0.1)
        self.damping: np.float32 = np.float32(0.25)
        self.contact_force: np.float32 = np.float32(1e2)
        self.contact_margin: np.float32 = np.float32(1e-3)
        
        self.collide_distance = self.agent_size * 2


        # Shit I don't know why but exists and I'l keep for now
        self.dim_c = 2


    def step(self):
        # n_agents long
        p_force: np.ndarray = np.zeros(shape=(self.agent_pos.shape[0],2))
        # assume all agents have actions
        p_force += self.agent_actions # they have a concept of noise, we don't
        # I believe ryan can do this

    def apply_environment_force(self, p_force: np.ndarray) -> None:
        pass

    

'''

    
 

