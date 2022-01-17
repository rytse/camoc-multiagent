import numpy as np
from gym import spaces
from gym.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
#from .._mpe_utils.core import Agent, Landmark, World
#from .._mpe_utils.scenario import BaseScenario

from itertools import combinations


class ComplexWorld(World):
    def __init__(self):
        super(ComplexWorld, self).__init__()
    
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        # integrate rotations
        for a, entity_a in enumerate(self.entities):
            entity_a.state.p_pos[2] += entity_a.state.p_vel[1] * self.dt
            orientation = entity_a.state.p_pos[2]
            v_x = entity_a.state.p_vel[0] * np.cos(orientation)
            v_y = entity_a.state.p_vel[0] * np.sin(orientation)
            
            v_x -= (1 - self.damping) * v_x
            v_y -= (1 - self.damping) * v_y
            if a == len(self.entities) - 1: continue
            for b, entity_b in enumerate(self.entities[a:]):
                f_a, f_b = self.get_collision_force(entity_a, entity_b)
                d = entity_a.state.p_pos[0] - entity_b.state.p_pos[0]
                v_x += f_a / entity_a.mass * self.dt
                v_y += f_a / entity_a.mass * self.dt


        
        # integrate velocities
        p_force = [None for _ in range(len(self.entities))]
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a):
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            

        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel[:2] = entity.state.p_vel[:2] * (1 - self.damping)
            if (acc_forces[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

        
        # check collisions

        
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)




    '''
    We need this one
    '''
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [0, 0]  # not a collider
        if (entity_a is entity_b):
            return [0, 0]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos[:2] - entity_b.state.p_pos[:2]
        dist = np.linalg.norm(delta_pos)
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else 0
        force_b = -force if entity_b.movable else 0
        return [force_a, force_b]

    # gather agent action forces
    def apply_action_force(self, p_force):
        raise Exception("apply_action_force was called!!!!")

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        raise Exception("apply_environment_force was called!!!!")

    # integrate physical state
    def integrate_state(self, p_force):
        raise Exception("integrate_state was called!!!!")

    



class ComplexSpreadScenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # world.dim_p = 3 # X Y Theta 
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        return world


    def _random_world_position(self, np_random):
        pos = np.zeros(shape=(3,))
        pos[:2] = np_random.uniform(-1,1, 2)
        pos[2] = np_random.uniform(-np.pi/180, np.pi/180,1)
        return pos



    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            #agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos = self._random_world_position(np_random)
            agent.state.p_vel = np.zeros(2)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            #landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_pos = self._random_world_position(np_random)
            landmark.state.p_vel = np.zeros()

    '''
    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)
    '''
    

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos[:2] - agent2.state.p_pos[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def global_reward(self, world):
        # TODO: IMPLEMENT GLOBAL REWARD PROPERLY
        rew = 0
        in_place_mesh = self.target_mesh - world.landmarks[0].state.p_pos[:2] 

        for a in world.agents:
            theta = a.state.p_pos[2]
            nose_pos = a.state.p_pos[:2] + a.size * np.array([np.cos(theta), np.sin(theta)])
            rew -= np.min(np.linalg.norm(in_place_mesh - nose_pos))

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(np.linalg.norm(entity.state.p_pos[:2] - agent.state.p_pos[:2]))
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(np.linalg.norm(other.state.p_pos[:2] - agent.state.p_pos[:2]))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)





class SimpleEnv(AECEnv):
    def __init__(self, scenario, world, max_cycles, continuous_actions=False, local_ratio=None):
        super().__init__()

        self.seed()

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.world.agents)}

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(low=0, high=1, shape=(space_dim,))
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(obs_dim,), dtype=np.float32)

        self.state_space = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(state_dim,), dtype=np.float32)

        self.steps = 0

        self.current_actions = [None] * self.num_agents

        self.viewer = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32)

    def state(self):
        states = tuple(self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32) for agent in self.possible_agents)
        return np.concatenate(states, axis=None)

    def reset(self):
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0. for name in self.agents}
        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._reset_render()

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.dones[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

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
            if np.all(other.state.c == 0):
                word = '_'
            elif self.continuous_actions:
                word = '[' + ",".join([f"{comm:.2f}" for comm in other.state.c]) + "]"
            else:
                word = alphabet[np.argmax(other.state.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()


