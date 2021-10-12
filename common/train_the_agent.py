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
    ##### parameters loading ####
    with open(args.solution_file, 'rb') as f:
        solutions = pickle.load(f)['solutions']

    state = envs.reset()
    #import ipdb; ipdb.set_trace()
    state = hwc2chw(state)
    current_state = torch.FloatTensor(np.float32(state))
    rollout.states[0].copy_(current_state)
    
    i_step = 0
    s = time.time()
    best_win_ratio = 0

    #if the agent should be trained until the performance plateau, we just simply set the number of training step to a really large number, which is 1e8.

    def get_current_potential():
        current_potential = expert.get_potentials(envs._get_room_states())
        #if the previous state is already in irreversible cases, we set potential = 0
        #print(current_potential)
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
            #action = manager.get_next_action(envs._get_room_states(), obs=Variable(current_state), agent=actor_critic)
            #action = torch.tensor(action)

            next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().detach().numpy())
            #next_state = next_state.reshape(num_envs, *state_shape)
            next_state = hwc2chw(next_state)

            reward = torch.FloatTensor(reward).unsqueeze(1)
            #reward shaping
            if i_step % args.ask_freq == 0:
                #expert_action = envs.get_astar_action()
                #dynamically set the bonus_reward;
                current_potential = get_current_potential()
                potential = current_potential - previous_potential
                #print('current potential: {}'.format(current_potential))
                #print('previous potential: {}'.format(previous_potential))
                time.sleep(0.5)
                print('potential: {}'.format(potential))
                
                #for satisfying \phi(S_0)=0
                potential[done==True] = 0

                '''
                current_potential = torch.zeros(len(action))
                current_potential[torch.eq(Variable(torch.tensor(expert_action.squeeze(2))), action).squeeze(1)] = args.bonus_reward
                #F(s,a,s',a') = gamma * Potential(s',a') - Potential(s,a)
                potential = args.gamma * current_potential - previous_potential
                '''
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
        # update the model for num_steps step, which is also the same step of the length of the rollout stotage
        ################# compute loss #################
        returns = rollout.compute_returns(next_value, args.gamma)
        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(Variable(rollout.states[:-1]).view(-1, *state_shape), Variable(rollout.actions).view(-1, 1))
        values = values.view(args.rolloutStorage_size, num_envs, 1)
        action_log_probs = action_log_probs.view(args.rolloutStorage_size, num_envs, 1)
        #advantage is the difference between predicted value and ground value
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
            #print(envs.dqm.dqm_dict)
            #calculate reward_actions/total_actions
            #rewarded_ratio = torch.eq(Variable(torch.tensor(expert_action.squeeze(2))), action).sum().item()/len(action)
            #wandb.log({'rewarded_ratio': rewarded_ratio})

            print('evaluating the model....')
            solved_rate, eval_avg_reward, avg_solved_l, avg_unsolved_l, n_solved, n_unsolved = test_the_agent(actor_critic, args.map_file, args.GPU, args.eval_num, display=args.display, args=args, Variable=Variable, solutions=solutions)
            wandb_session.log({'solved_ratio': solved_rate})

            e = time.time()

            printer_stuff = {
                            'solved rate over {} games'.format(args.eval_num): solved_rate,
                            'average solved solutions over {} games'.format(args.eval_num): avg_solved_l,
                            'solved maps over {} games'.format(args.eval_num): n_solved,
                            'average unsolved solutions over {} games'.format(args.eval_num): avg_unsolved_l,
                            'unsolved maps over {} games'.format(args.eval_num): n_unsolved,
                            #'total trained env steps': actor_critic.steps_done,
                            'total trained env steps': i_step,
                            'average evaluated rewards over {} games'.format(args.eval_num): eval_avg_reward,
                            'training on': args.map_file,
                            'costed time so far': e - s
                            }
            print(printer_stuff)

            #plt.plot([i for i in range(len(eval_avg_reward))], eval_avg_reward)
            #display.display(plt.gcf())
            #display.clear_output(wait=True)
            
    #if the total number reaches the threshold but still doesnt solve the level, we return -1
    envs.close()
