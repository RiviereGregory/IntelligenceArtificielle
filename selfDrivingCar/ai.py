

import numpy as np
import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, inputSize, nbAction):
        super(Network, self).__init__()
        self.inputSize = inputSize
        self.nbAction = nbAction
        self.fc1 = nn.Linear(inputSize, 30) # création reseau neurone caché entrée
        self.fc2 = nn.Linear(30, nbAction) # création reseau neurone caché sortie
        
    def forward(self, state):
        x = F.relu(self.fc1(state)) # couche cachée
        qValues = self.fc2(x) # Prédiction
        return qValues
    
# Expèrience replay
        
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batchSize):
        samples = zip(*random.sample(self.memory, batchSize))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning
        
class Dqn():
    
    def __init__(self, inputSize, nbAction, gamma):
        self.model = Network(inputSize, nbAction)
        self.gamma = gamma
        self.reward_window = []        
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(inputSize).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

