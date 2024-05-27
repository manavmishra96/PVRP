import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import os

from net import ActorCritic, Memory

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    
class PPO:
    def __init__(self, env, num_agents):
        self.lr = 0.000002
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.len_episode = 1000000
        
        self.num_agents = num_agents
        
        torch.manual_seed(11)
        
        self.policy = ActorCritic(env, self.num_agents).to(device)
#         self.loadModel(filePath='checkpoints/ActorCritic_5600.pt')
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        # Needed for the clipped objective function
        self.policy_old = ActorCritic(env, self.num_agents).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        idx = 0
        while os.path.exists(f"tf_log/demo_{idx}"):
            idx = idx + 1   
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_{idx}")
        print(f"Log Dir: {self.sw.log_dir}")
        
    def change_num_agents(self, num_agents):
        self.num_agents = num_agents
        self.policy.change_num_agents(num_agents)
        self.policy_old.change_num_agents(num_agents)
      
    # back prop  
    def update(self, memory):
        all_rewards = []
        discounted_reward_list = [0] * int(self.num_agents)
        agent_index_list = list(range(self.num_agents)) * int(len(memory.rewards)/self.num_agents)
        for reward, is_terminal, agent_index in zip(reversed(memory.rewards), reversed(memory.is_terminals), reversed(agent_index_list)):
            if is_terminal:
                discounted_reward_list[agent_index] = 0
            discounted_reward_list[agent_index] = reward + (self.gamma * discounted_reward_list[agent_index])
            all_rewards.insert(0, discounted_reward_list[agent_index])

        all_rewards = torch.tensor(all_rewards).to(device)
        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        
        minibatch_sz = self.num_agents * self.len_episode
        
            
        mem_sz = len(memory.states)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            prev = 0
            for i in range(minibatch_sz, mem_sz+1, minibatch_sz):
                mini_old_states = memory.states[prev:i]
                mini_old_actions = memory.actions[prev:i]
                mini_old_logprobs = memory.logprobs[prev:i]
                mini_rewards = all_rewards[prev:i]
                
                # Convert list to tensor
                old_states = torch.stack(mini_old_states).to(device).detach()
                old_actions = torch.stack(mini_old_actions).to(device).detach()
                old_logprobs = torch.stack(mini_old_logprobs).to(device).detach()
                rewards = mini_rewards #torch.from_numpy(mini_rewards).float().to(device)
                
                prev = i
                # Evaluating old actions and values :
                logprobs, state_values = self.policy.evaluate(old_states, old_actions)
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs.view(-1,1) - old_logprobs.view(-1,1).detach())
                    
                # Finding Surrogate Loss:
                advantages = (rewards - state_values.detach()).view(-1,1)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean()

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, rewardHistory, lenEpisode):
        self.sw.add_scalar('3.Reward', rewardHistory[-1], episode)
        self.sw.add_scalar('2.Episode Length', lenEpisode, episode)
        
        if len(rewardHistory)>=100:
            avg_reward = mean(rewardHistory[-100:])
        else:    
            avg_reward = mean(rewardHistory) 
        self.sw.add_scalar('1.Average of Last 100 episodes', avg_reward, episode)
            
    def summaryWriter_close(self):
        self.sw.close()
        
    def saveModel(self, filePath, per_save=False, episode=0):
        if per_save == False:
            torch.save(self.policy.state_dict(), f"{filePath}/{self.policy.__class__.__name__}.pt")
        else:
            torch.save(self.policy.state_dict(), f"{filePath}/{self.policy.__class__.__name__}_{episode}.pt")
    
    def loadModel(self, filePath, cpu = 0):
        if cpu == 1:
            self.policy.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
        else:
            self.policy.load_state_dict(torch.load(filePath))
        self.policy.eval()