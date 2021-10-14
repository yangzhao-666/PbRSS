import gym
import gym_sokoban

import torch
import numpy as np
import random
import time

from utilities.channelConverter import hwc2chw
from experts.utils import get_distance
from external_actions import get_astar_action

import warnings
warnings.simplefilter("ignore", UserWarning)

def test_the_agent(agent, data_path, USE_CUDA, eval_num, args=None, display=False, deter=False, Variable=None):
    solved = []
    rewards = []
    
    #specify the environment you wanna use; v0 means sample sub-cases randomly, and v1 only sample targeted sub-cases;
    #env = gym.make('Curriculum-Sokoban-v2', data_path = data_path, seed=random.randint(0,100))
    env = gym.make('Curriculum-Sokoban-v2', data_path = data_path)

    solved_maps = []
    unsolved_maps = []
    for i in range(eval_num):

        episode_reward = 0

        state = env.reset()
        if display:
            print('#### Start ####')
            print(env.room_state)
            print('{} steps towards the goal state'.format(get_distance(env.room_state)))
            time.sleep(1)
        state = hwc2chw(state, test=True)
        if USE_CUDA:
            state = state.cuda()
        action = agent.select_action(state.unsqueeze(0), test=1, determinisitc=deter)
        next_state, reward, done, _ = env.step(action.item())
        episode_reward += reward
        next_state = hwc2chw(next_state, test=True)
        if display:
            print('#### action taken ####')
            print('taken action is {}, expert action is {}'.format(action.item(), get_astar_action(env.room_state)))
            print(env.room_state)
            print('{} steps towards the goal state'.format(get_distance(env.room_state)))
            time.sleep(1)

        i = 1

        while not done:
            state = next_state
            if USE_CUDA:
                state = state.cuda()
            with torch.no_grad():
                action = agent.select_action(state.unsqueeze(0), test=1, determinisitc=deter)
            if display:
                print('#### action taken ####')
                print('taken action is {}, expert action is {}'.format(action.item(), get_astar_action(env.room_state)))
                print(env.room_state)
                print('{} steps towards the goal state'.format(get_distance(env.room_state)))
                time.sleep(1)
            next_state, reward, done, _ = env.step(action.item())
            if get_distance(env.room_state) == -1:
                if display:
                    print('The game is unsolvable now')
                time.sleep(2)
                break

            episode_reward += reward
            next_state = hwc2chw(next_state, test=True)

            i += 1
        if i < env.max_steps and get_distance(env.room_state) != -1:
            solved.append(1)
            solved_maps.append(env.selected_map)
        else:
            unsolved_maps.append(env.selected_map)
        
        rewards.append(episode_reward)

    return np.sum(solved)/eval_num
