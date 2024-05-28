import os
import argparse
import numpy as np
import json

from tqdm import tqdm
import cv2
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from time import time as t
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from net import Memory
from ppo import PPO
import pickle
import warnings
from env import PVRP

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)
warnings.filterwarnings('ignore')

NUM_EPISODES = 2000000
LEN_EPISODES = 100
UPDATE_TIMESTEP = 100


curState = []
newState= []
reward_history = []
agent_history_dict = defaultdict(list)
totalViewed = []
dispFlag = False
keyPress = 0
timestep = 0
loss = None
memory = Memory()

directory = 'test'
print(directory)

if not os.path.exists(directory):
    os.makedirs(directory)
    os.mkdir(f"{directory}/checkpoints")




def mask_fn(env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env.
    masks = []
    for agent in env.agents:
        mask = [True]*(env.n + 1) + [False]*(env.N - env.n)
        mask[agent.loc] = False
    
        ori_term = agent.dist + np.linalg.norm(env.locations - [0, 0], axis = 1)
        mask = np.logical_and(mask, ori_term <= agent.fuel)
        masks.append(mask)
        
    return np.array(masks)

NUM_AGENTS = 3
N = 10
env = PVRP(N, NUM_AGENTS)
RL = PPO(env, NUM_AGENTS)


track_rewards = []
for episode in tqdm(range(NUM_EPISODES)):
    cur_num_agents = NUM_AGENTS
    curRawState = env.reset()
    
    curState = curRawState
    
    episodeReward  = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    agent_episode_reward = [0] * cur_num_agents
    
    for step in range(LEN_EPISODES):
        timestep += 1

        masks = mask_fn(env)
        masks = masks.astype(int)
        action = RL.policy_old.act(curState, memory, cur_num_agents, masks)
        newState  = env.step(action)
        states, reward, terminated, truncated, info = newState
        done = terminated or truncated
        
        if step == LEN_EPISODES - 1:
            done = True
        
        for agent_index in range(cur_num_agents):
            memory.rewards.append(float(reward))
            memory.is_terminals.append(done)
            
        
        #back prop 
        # check time termination condition
        if timestep % UPDATE_TIMESTEP == 0:
            RL.update(memory)
            memory.clear_memory()
            timestep = 0
        
        # record history
        for i in range(cur_num_agents):
            agent_episode_reward[i] += reward
            
        episodeReward += reward

        # set current state for next step
        curState = newState[0]
        
        if done:
            track_rewards.append(episodeReward)
            break
        
    # post episode
    
    reward_history.append(episodeReward)
    
    
    RL.summaryWriter_addMetrics(episode, reward_history, LEN_EPISODES)
    if episode % 50 == 0:
        RL.saveModel(directory+"/checkpoints")

    if episode % 100 == 0:
        RL.saveModel(directory+"/checkpoints", True, episode)
            
    
RL.saveModel(directory+"/checkpoints")


# Calculate the moving average
moving_avg = np.convolve(track_rewards, np.ones((200,))/200, mode='valid')

# Calculate the standard deviation
std_dev = np.std(track_rewards)

# Plot the moving average
plt.plot(moving_avg)

# Fill the area between the moving average and the standard deviation
plt.fill_between(range(len(moving_avg)), moving_avg - std_dev, moving_avg + std_dev, color='b', alpha=0.1)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Track Rewards')
plt.savefig('track_rewards.png')