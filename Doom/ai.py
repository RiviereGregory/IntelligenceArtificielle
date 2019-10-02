#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym import wrappers
import vizdoomgym

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

#Cerveau
class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)
        
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution2(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution3(x), kernel_size=3, stride=2))
        return x.data.view(1, -1).size(1)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution2(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution3(x), kernel_size=3, stride=2)) 
        x = x.view(x.size(0), -1) # Flatening
        x = F.relu(self.fc1(x)) # Fonction d'activation
        x = self.fc2(x)
        return x
#Corps
class SoftmaxBody(nn.module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
          
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=0)
        actions = probs.multinomial(num_samples=1)
        return actions
    
# liens entre corps et cerveau
class AI:
    
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        
    def __call__(self, inputs):
        inputs = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        outputs = self.brain(inputs)
        actions = self.body(outputs)
        return actions.data.numpy        
        


# Part 2 - Training the AI with Deep Convolutional Q-Learning

