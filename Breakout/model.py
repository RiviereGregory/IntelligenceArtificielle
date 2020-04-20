#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# AI for Breakout

# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialisation des poids de manière optimale (std --> variance)
def normalized_columns_initializer(size, std):
    out = torch.randn(size)
    out *= std / torch.sqrt(out.pow(2).sum(1, True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = m.weight.data.size()   
        fan_in = np.prod(weight_shape[1:4]) 
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = m.weight.data.size()   
        fan_in = weight_shape[1] # nombre de neurone en entrée
        fan_out = weight_shape[0] # nombre de neurone en sortie
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        
# Fabrication du cerveau de L'IA
class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        # Création des oeils (couvhe de neurone)
        # taille image d'entree 42*42
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # taille image 21*21
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # taille image 12*12
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # taille image 6*6
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # taille image 3*3
        # couche de entieremetn connecte
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256) # nombre d'image * par la taille image, nombre neurone sortie
        num_outputs = action_space.n
        # Critic
        self.critic_linear = nn.Linear(256, 1)
        # Actor
        self.actor_linear = nn.Linear(256, num_outputs)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data.size(), 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data.size(), 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()
        
        
    def forward(self, inputs):
        inputs, (hx, cx ) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        # flattening
        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        