import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env


class Agent():
    def __init__(self, loc=0, max_fuel=100, max_steps=100, id=0):
        self.id = id
        self.loc = loc
        self.max_steps = max_steps
        self.max_fuel = max_fuel
        self.fuel = self.max_fuel
        self.dist = []
        
    def update_fuel(self, distance):
        self.fuel -= distance
        
    def refuel(self):
        self.fuel = self.max_fuel


class PVRP(gym.Env):
    def __init__(self, N, num_agent, n = None, w=None, poses=None, seed=None, max_fuel = 100, max_time = 200):
        super(PVRP, self).__init__()
        if seed is not None:
            np.random.seed(seed=seed)
        
        self.steps = 0
        self.N = N
        self.T = 200
        
        self.max_fuel = 100
        self.max_steps = 100
        self.agents = [Agent(max_fuel=self.max_fuel, max_steps=self.max_steps, id=i) for i in range(num_agent)]
        
        if n is not None:
           self.n = n
           
        else: 
            self.n = np.random.randint(2, self.N + 1)
        
        if w is not None:
            self.w = w
        else:
            self.w = np.zeros(self.N + 1)
            self.w[:self.n + 1] = 1.0
        
        if poses is not None:
            self.locations = poses
        else:
            self.locations = self._generate_points(self.n)
        # print(f'locations: {self.locations}')    
        self.distances = self._calculate_distances(self.locations)
        # print(f'distances: {self.distances}')
        
        self.observation_space = 1 + self.N + 1 + self.N + 1 + 2*(self.N + 1) + 1
        self.action_space = self.N + 1
    
    
    def reset(self, n=None, w=None, poses=None):
        self.steps = 0
        self.fuel = self.max_fuel
        
        # if n is not None:
        #     self.n = n
        # else:
        #     self.n = np.random.randint(2, self.N)
        
        # if w is not None:
        #     self.w = w
        # else:
        #     self.w = np.zeros(self.N + 1)
        #     self.w[:self.n + 1] = 1.0 #+ np.round(np.random.uniform(size = self.n + 1), 2)
        
        # if poses is not None:
        #     self.locations = poses
        # else:
        #     self.locations = self._generate_points(self.n)
            
        # self.distances = self._calculate_distances(self.locations)
        
        for agent in self.agents:
            agent.loc = 0
            agent.dist = self.distances[agent.loc]
            
        self.clocks = np.zeros(self.N + 1)
            
        states = self._get_states()
        return states
    
    
    def reset_(self, n=None, w=None, poses=None):
        self.steps = 0
        self.fuel = self.max_fuel
        
        if n is not None:
            self.n = n
        else:
            self.n = np.random.randint(2, self.N + 1)
        
        if w is not None:
            self.w = w
        else:
            self.w = np.zeros(self.N + 1)
            self.w[:self.n + 1] = 1.0
        
        if poses is not None:
            self.locations = poses
        else:
            self.locations = self._generate_points(self.n)
            
        self.distances = self._calculate_distances(self.locations)
        
        for agent in self.agents:
            agent.loc = 0
            agent.dist = self.distances[agent.loc]
            
        self.clocks = np.zeros(self.N + 1)
        
        states = self._get_states
        return states
    
    
    def step(self, action):
        self.steps += 1
        for agent, next_loc in zip(self.agents, action):
            self.clocks += self.w * self.distances[agent.loc, next_loc]
            self.clocks = np.clip(self.clocks, 0, self.T)
            dist_travelled = self.distances[agent.loc, next_loc]
            agent.update_fuel(dist_travelled)
            agent.dist = self.distances[next_loc]

        reward = self._get_rewards(self.clocks)
        for next_loc in action:
            self.clocks[next_loc] = 0
        
            
        terminated = (self.steps == self.max_steps)
        truncated = any(agent.fuel < 0 for agent in self.agents)
        
        for agent, next_loc in zip(self.agents, action):
            if next_loc == 0:   # depot condition
                agent.refuel()
            
        next_state = self._get_states()
        return next_state, reward, terminated, truncated, {}
    
    
    def _generate_points(self, n):
        points = [[0,0]]
        # Generate n random 2D points within the 10x10 grid
        # for _ in range(n):
        #     x = np.random.randint(1, 10)
        #     y = np.random.randint(1, 10)
          
            # else:
            #     _ -= 1
        while len(points) <= n:
            x = np.random.random() * 10
            y = np.random.random() * 10
            # if [x, y] not in points:
            # points.append([x, y])
            if [x, y] not in points:
                points.append([x, y])
                
        # Fill the remaining N-n points with (0,0)
        remaining = self.N - n
        for _ in range(remaining):
            points.append([0, 0])
            
        return np.array(points)    
    
    
    def _calculate_distances(self, locations):
        n = len(locations)
        
        distances = np.zeros((n, n))    
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
                
        return distances
    
    
    def _get_rewards(self, next_clocks):
        # if next_loc == self.loc or distance == 0:
        #     return -1000
        
        # if next_fuel > 0:
        reward = -np.max(next_clocks[1:])
        # else:
        #     # reward = 150 * math.floor(next_fuel)
        #     reward = -150 * (self.max_steps - self.steps)
        return reward
    
    
    def _get_states(self):
        states = []
        for agent in self.agents:
            state = np.concatenate((np.array([agent.loc]), np.array(self.clocks), np.array(agent.dist), np.array(self.locations).reshape(-1), np.array([agent.fuel])), dtype=np.float32)
            states.append(state)
        return np.array(states)
    
    
    def render_(self):
        pass
    
    


        
if __name__ == '__main__':
    N = 10
    num_agent=3
    
    env = PVRP(N, num_agent)
    obs = env.reset()
    print(obs[0].shape)
    
    for _ in range(100):
        action = [np.random.randint(0, N) for _ in range(num_agent)]
        obs_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            break
        
        # print(f"Step: {env.steps}, Current location: {env.loc}, Current clock readings: {env.clocks}, Current distances: {env.dist}, Fuel left: {env.fuel}")
        # print(f"Reward = {reward}")
        # print(f"targets: {env.n}")
        obs = obs_
        
    # check_env(env, warn=True)