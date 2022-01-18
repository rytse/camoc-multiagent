from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.env import AECEnv
import numpy as np
from gym.utils import seeding
from typing import Optional, int
from itertools import combinations
from scipy.spatial import distance_matrix

class FastSimpleEnv(AECEnv):
    def __init__(self, scenario, world, max_cycles, continuous_actions=False, local_ratio=None):
        super().__init__()
        self.seed()
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)
        
        self.action_spaces = None 
        self.observation_spaces = None

        # Initialize spaces is important 

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

class raw_env(FastSimpleEnv):
    def __init__(self, N=5, local_ratio=0.25, max_cycles=25, continuous_actions=True):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = FastScenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata['name'] = "swarm_cover_v1"




# Could inherit base Scenario
class FastScenario:
    def make_world(self, N=3, AGENT_SIZE=0.15, LANDMARK_SIZE=1.5, N_THETA=10):
        world = FastWorld(N, 1, AGENT_SIZE, LANDMARK_SIZE)
        # set any world properties first
        #world.collaborative = True

        # add agents

        # add landmarks

        # Generate mesh that approximates the target
        theta = np.linspace(0, 2*np.pi, N_THETA)
        xx = LANDMARK_SIZE * np.cos(theta)
        yy = LANDMARK_SIZE * np.sin(theta)
        self.target_mesh = np.rollaxis(np.array([xx, yy]), 1)

        return world

    def global_reward(self, world):
        rew = 0

        '''
        agent_thetas_relative_to_landmark = []
        for a in world.agents:
            rel_x,rel_y = a.pos - world.landmarks[0].pos
            theta_lm = np.arctan2(rel_y, rel_x)
            agent_thetas_relative_to_landmark.append(theta_lm)

            nose_pos = a.pos[:2] + a.size * np.array([np.cos(a.theta), np.sin(a.theta)])
            rew -= np.min(np.linalg.norm(in_place_mesh - nose_pos))

        # We want to encourage them to spread out
        # Ideally all agents equally spaced from each each other
        # with first and last agents 
        
        # we want the largest spread possible
        variance = np.var(agent_thetas_relative_to_landmark)
        variance += np.float(1e-6) # blowing up is bad
        rew /= variance
        '''
        

        return rew
    

    def reset_world(self, world, np_random):
        world.reset(np_random)


    def observation(self, agent_index: int, world) -> np.ndarray:
        return world.observation(self, agent_index)
        





class FastWorld:
    def __init__(self, n_agents: np.int, n_entities: np.int, entity_sizes: np.ndarray):
        self.n_agents = n_agents
        self.n_entities = n_entities
        self.entity_sizes = entity_sizes

        self.positions: np.ndarray = np.zeros(shape=(n_entities, 2))
        self.velocities: np.ndarray = np.zeros(shape=(n_entities, 2))

        self.ctrl_thetas: np.ndarray = np.zeros(n_entities)
        self.ctrl_speeds: np.ndarray = np.zeros(n_entities)

        # Only agents are movable
        self.movables = np.zeros(n_entities)
        self.movables[:n_agents] = 1

        self.entity_sizes: np.ndarray = entity_sizes

        self.sizemat = self.entity_sizes[..., None] * self.entity_sizes[None, ...]
        

        # World parameters
        self.dt = 0.15
        self.damping = 0.1

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
        entity_pos: np.ndarray = self.landmarks - self.agents[agent_index]
        other_pos = np.delete(self.agents, agent_index)
        other_pos -= self.agents[agent_index]
        return np.concatenate(self.velocities[agent_index] + [self.agents[agent_index]] + entity_pos + other_pos)
        


    def reset(self, np_random) -> None:
        """
        Resets the world
        """
        self.positions =  np_random.unform(-1, +1, shape=(self.positions.shape))
        self.velocities[:] = 0
        


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

    
 

