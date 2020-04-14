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
    out = out / torch.sqrt(out.pow(2).sum(1, keepdim=True)) * std
    return out
