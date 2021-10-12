'''
@Author: Michiel and Zhao
@Date: 2021.3.31
@Description: functions are for getting an action given a state and an agent, will return the next action; you can extend it by implementing your own 'expert' by taking states/obs, agent/model/search algorithm as inputs, and output the next action according to your input, and then call the function in Manager.py;
'''
from experts.sokoban import get_solution
from experts.utils import check_next_irreversible
from experts.utils import get_distance

import random
import numpy as np

def get_dumb_action(state):
    #return random action;
    actions = [0, 1, 2, 3, 4]
    action = random.choice(actions)
    return [[action]]

def get_half_expert_action(state):
    #return the action by the expert who makes mistakes sometimes;
    raise NotImplementedError

def get_human_action(state):
    #return the action by real human who might know or might not know the solution;
    #will need to show the game board to game and wait for the input;
    raise NotImplementedError

def get_worst_action(state):
    #it just doesn't give you the optimal/right actions;
    raise NotImplementedError

def get_irreversible_action(state):
    if check_next_irreversible(state):
        return get_astar_action(state)
    else:
        return False

def get_astar_action(state):
    #return the optimal action came from A* search;
    solution = get_solution(state, 'astar')
    if len(solution)<1:
        print('debug info: solution is {}'.format(solution))
        raise ValueError('the len of solution is smaller than 1')
    action = solution[0]
    if action == 'u' or action == 'U':
        action = 1
    elif action == 'd' or action == 'D':
        action = 2
    elif action == 'l' or action == 'L':
        action = 3
    elif action == 'r' or action == 'R':
        action = 4
    else:
        action = -1
    return [[action]]

def get_dfs_action(state):
    #return the optimal action came from Depth First Search;
    solution = get_solution(state, 'dfs')
    action = solution[0]
    if action == 'u' or action == 'U':
        action = 1
    elif action == 'd' or action == 'D':
        action = 2
    elif action == 'l' or action == 'L':
        action = 3
    elif action == 'r' or action == 'R':
        action = 4
    else:
        action = 0
    return [[action]]

def get_bfs_action(state):
    #return the optimal action came from Breath First Search;
    solution = get_solution(state, 'bfs')
    action = solution[0]
    if action == 'u' or action == 'U':
        action = 1
    elif action == 'd' or action == 'D':
        action = 2
    elif action == 'l' or action == 'L':
        action = 3
    elif action == 'r' or action == 'R':
        action = 4
    else:
        action = 0
    return [[action]]

def get_ucs_action(state):
    #return the optimal action came from Uniform Cost Search;
    solution = get_solution(state, 'ucs')
    action = solution[0]
    if action == 'u' or action == 'U':
        action = 1
    elif action == 'd' or action == 'D':
        action = 2
    elif action == 'l' or action == 'L':
        action = 3
    elif action == 'r' or action == 'R':
        action = 4
    else:
        action = 0
    return [[action]]

def get_starter_action(state, threshold):
    dis = get_distance(state)
    if dis > threshold:
        solution = get_solution(state, 'astar')
        action = solution[0]
        if action == 'u' or action == 'U':
            action = 1
        elif action == 'd' or action == 'D':
            action = 2
        elif action == 'l' or action == 'L':
            action = 3
        elif action == 'r' or action == 'R':
            action = 4
        else:
            action = 0
        return [[action]]
        
    else:
        return False

def get_terminator_action(state, threshold):
    dis = get_distance(state)
    if dis < threshold:
        solution = get_solution(state, 'astar')
        action = solution[0]
        if action == 'u' or action == 'U':
            action = 1
        elif action == 'd' or action == 'D':
            action = 2
        elif action == 'l' or action == 'L':
            action = 3
        elif action == 'r' or action == 'R':
            action = 4
        else:
            action = 0
        return [[action]]
        
    else:
        return False

def get_meta_action(state, dis, meta_expert):
    yes_or_no = meta_expert(dis.reshape(-1, 1))
    pro_0 = yes_or_no[:, 1].cpu().detach().numpy().astype('float64')
    pro_1 = 1 - pro_0
    actions = []
    i = 0
    for p_0, p_1 in zip(pro_0, pro_1):
        choice = np.random.choice([0, 1], p=[p_0, p_1])
        if choice == 1:
            solution = get_solution(state[i], 'astar')
            action = solution[0]

            if action == 'u' or action == 'U':
                action = 1
            elif action == 'd' or action == 'D':
                action = 2
            elif action == 'l' or action == 'L':
                action = 3
            elif action == 'r' or action == 'R':
                action = 4
            else:
                action = 0
        else:
            action = None
        i += 1
        actions.append(action)
    return actions
