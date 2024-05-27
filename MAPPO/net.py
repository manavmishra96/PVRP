import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
import warnings
from env import PVRP

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


    
class ActorCritic(nn.Module):
    def __init__(self, env, init_num_agents):
        super(ActorCritic, self).__init__()
        
        self.reg1 = nn.Sequential(
            nn.Linear(env.observation_space, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space),
            # nn.Softmax(dim=-1)
        )
        
        self.reg2 = nn.Sequential(
            nn.Linear(env.observation_space, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.train()
            
            
    def action_layer(self, x, masks):
        masks = torch.tensor(masks, dtype=torch.float32).to(device)
        logits = self.reg1(x)
        masked_logits = logits + (1 - masks) * -1e8
        action_probs = F.softmax(masked_logits, dim=-1)
        return action_probs
    
    
    def value_layer(self, x):
        x = self.reg2(x)
        return x
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = (torch.diag(dist.log_prob(action))).view(-1,1)
        
        state_value = self.value_layer(state)
        return action_logprobs, torch.squeeze(state_value)
    
    
    def act(self, state, memory, num_agents, masks):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state, masks)
            dist = Categorical(action_probs)
            num = np.random.uniform()
            action = dist.sample()
                
            action_list = []
            for agent_index in range(num_agents):
                memory.states.append(state[agent_index])
                memory.actions.append(action[agent_index].view(1))
                memory.logprobs.append(dist.log_prob(action[agent_index])[agent_index])
                action_list.append(action[agent_index].item())
        return action_list


    def act_max(self, state, memory, num_agents, masks):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state, masks)
            dist = Categorical(action_probs)
            action = torch.argmax(action_probs, dim=1)
                
            action_list = []
            for agent_index in range(num_agents):
                memory.states.append(state[agent_index])
                memory.actions.append(action[agent_index].view(1))
                memory.logprobs.append(dist.log_prob(action[agent_index])[agent_index])
                action_list.append(action[agent_index].item())
        return action_list