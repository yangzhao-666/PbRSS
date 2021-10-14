import gym
import gym_sokoban

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle
import time

from utilities.channelConverter import hwc2chw
from common.test_the_agent import test_the_agent

import warnings
warnings.simplefilter("ignore", UserWarning)

def train_the_agent(expert, envs, num_envs, Variable, state_shape, actor_critic, optimizer, rollout, args, wandb_session):

    state = envs.reset()
    state = hwc2chw(state)
    current_state = torch.FloatTensor(np.float32(state))
    rollout.states[0].copy_(current_state)
    
    i_step = 0
    s = time.time()
    best_win_ratio = 0

    def get_current_potential():
        current_potential = expert.get_potentials(envs._get_room_states())
        #if the previous state is already in irreversible cases, we set potential = 0
        for idx in range(args.num_envs):
            if current_potential[idx] == 1: #this is the same as distance==-1
                if locker[idx]:
                    current_potential[idx] = previous_potential[idx]
                else: #if the current state just fall into irreversible case, we give it a -1 penalty.
                    current_potential[idx] = previous_potential[idx] - 1
                    locker[idx]  = True
            else:
                locker[idx] = False
        return current_potential


    while i_step < args.num_steps:
        
        for step in range(args.rolloutStorage_size):
            i_step += 1
            
            action = actor_critic.select_action(Variable(current_state))

            next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().detach().numpy())
            next_state = hwc2chw(next_state)

            reward = torch.FloatTensor(reward).unsqueeze(1)
            #reward shaping
            current_potential = get_current_potential()
            potential = current_potential - previous_potential
            time.sleep(0.5)
            print('potential: {}'.format(potential))
            
            potential[done==True] = 0

            reward += potential.cpu().detach()
            if args.display:
                print('reward: {}'.format(reward))
            previous_potential = current_potential
            masks = torch.FloatTensor(1-done).unsqueeze(1)
            if args.GPU:
                masks = masks.cuda()
            current_state = torch.FloatTensor(np.float32(next_state))
            rollout.insert(step, current_state, action.data, reward, masks)

        _, next_value = actor_critic(Variable(rollout.states[-1]))
        next_value = next_value.data
        ################# compute loss #################
        returns = rollout.compute_returns(next_value, args.gamma)
        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(Variable(rollout.states[:-1]).view(-1, *state_shape), Variable(rollout.actions).view(-1, 1))
        values = values.view(args.rolloutStorage_size, num_envs, 1)
        action_log_probs = action_log_probs.view(args.rolloutStorage_size, num_envs, 1)
        advantages = Variable(returns) - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        optimizer.zero_grad()
        loss = value_loss * args.value_loss_coef + action_loss - entropy * args.entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()
        rollout.after_update()
        
        if i_step % args.eval_per == 0:

            solved_rate = test_the_agent(actor_critic, args.map_file, args.GPU, args.eval_num, display=args.display, args=args, Variable=Variable)
            wandb_session.log({'solved_ratio': solved_rate})
            print('solved ratio: {}'.format(solved_rate))

    envs.close()
