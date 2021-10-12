'''
@Author: Zhao
@Date: 2021.05.29
@Description: train a2c agent with the help of experts
'''
import gym
import gym_sokoban

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import copy
import random
import pickle
import time

from utilities.channelConverter import hwc2chw
from utilities.printer import Printer
from utilities.logger import Logger
from common.ActorCritic import ActorCritic
from common.RolloutStorage import RolloutStorage
from common.cpu_or_gpu import cpu_or_gpu
from common.multiprocessing_env import SubprocVecEnv
from common.train_the_agent import train_the_agent
from expert import Expert

def train(args, wandb_session):

    ##################### logger, printer and other initial settings ################
    #env_seed = [random.randint(0, 100) for i in range(args.num_envs)]

    def make_env():
        def _thunk():
            #env = gym.make('Curriculum-Sokoban-v2', data_path = args.map_file, seed = i)
            env = gym.make('Curriculum-Sokoban-v2', data_path = args.map_file)
            return env
        return _thunk

    ##################### initialize agent, optimizer etc #########################
    #################### import for env, either nomal one or the one with early termination.
    #env_list = [make_env(i) for i in env_seed]
    env_list = [make_env() for i in range(args.num_envs)]
    envs = SubprocVecEnv(env_list)
    state_shape = (3, 80, 80)
    
    #number of action was 9, but the number of pushing action space is much smaller than moving space, so here we will try pushing action space which is 4+1=5
    #num_actions = envs.action_space.n
    num_actions = 5
    actor_critic = ActorCritic(state_shape, num_actions=num_actions, normalization=args.normalization)
    expert_actor_critic = ActorCritic(state_shape, num_actions=num_actions, normalization=args.normalization) #model for loading the pre-trained model to give q-values
    rollout = RolloutStorage(args.rolloutStorage_size, args.num_envs, state_shape)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    #according to the args to decide where the models are, gpu or cpu?
    Variable, actor_critic, rollout = cpu_or_gpu(args.GPU, actor_critic, rollout)

    ##################### load existing model #####################
    if args.mode == 'scratch':
        from common.train_the_agent_scratch import train_the_agent
        train_the_agent(envs, args.num_envs, Variable, state_shape, actor_critic, optimizer, rollout, args, wandb_session) #train and save the model;
    else:
        #initialize the expert
        expert = Expert(mode=args.mode, pre_trained_path=args.pre_trained_path, expert_model=expert_actor_critic, normalization=args.normalization)
        from common.train_the_agent import train_the_agent
        train_the_agent(expert, envs, args.num_envs, Variable, state_shape, actor_critic, optimizer, rollout, args, wandb_session) #train and save the model;
