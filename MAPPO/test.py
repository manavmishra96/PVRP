import torch
from tqdm import tqdm
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

memory = Memory()

NUM_EPISODES = 10
LEN_EPISODES = 100
UPDATE_TIMESTEP = 100

NUM_AGENTS = 3
N = 10
env = PVRP(N, NUM_AGENTS)
RL = PPO(env, NUM_AGENTS)

RL.loadModel("MAPPO/test/checkpoints/ActorCritic.pt")

for episode in range(NUM_EPISODES):
    print(f"Episode: {episode}")
    cur_num_agents = NUM_AGENTS
    curRawState = env.reset()
    
    curState = curRawState
    
    episodeReward  = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    agent_episode_reward = [0] * cur_num_agents
    
    for step in range(LEN_EPISODES):
        masks = mask_fn(env)
        masks = masks.astype(int)
        action = RL.policy_old.act(curState, memory, cur_num_agents, masks)
        newState  = env.step(action)
        states, reward, terminated, truncated, info = newState
        done = terminated or truncated
        
        print(f"Step: {env.steps}, Current clock readings: {env.clocks}")
        for i, agent in enumerate(env.agents):
            print(f"Agent {i+1} - Fuel left: {agent.fuel}, Location: {agent.loc}")
            
        episodeReward += reward

        # set current state for next step
        curState = newState[0]
        
        if done:
            break



