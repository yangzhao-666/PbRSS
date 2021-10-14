import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from .OnPolicy import OnPolicy

import gym
import gym_sokoban

class ActorCritic(OnPolicy):
    
    def __init__(self, state_shape, num_actions, normalization=True):
        super(ActorCritic, self).__init__()

        self.normalization = normalization #whether normalize pre-trained value functions
        self.state_shape = state_shape
        self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(self.feature_size(), 512),
                nn.ReLU(),
                )
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

    def forward(self, x):
#import ipdb; ipdb.set_trace()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.state_shape))).view(1, -1).size(1)

    def get_critic(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        value = self.critic(x)
        if self.normalization:
            value = value/torch.sum(value)
        return value
