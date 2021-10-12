import json
import os
import pickle

from train import train

import argparse
import wandb

def main():
    description = 'Sokoban LfRS'
    parser = argparse.ArgumentParser(description=description)
    #training settings
    parser.add_argument('--num_steps', type=int, default=150000)
    parser.add_argument('--num_envs', type=int, default=30)
    parser.add_argument('--eval_per', type=int, default=1000)
    parser.add_argument('--eval_num', type=int, default=20)
    parser.add_argument('--GPU', type=bool, default=True)
    #files settings
    parser.add_argument('--map_file', type=str, default='../maps/3_boxes/')
    parser.add_argument('--solution_file', type=str, default='../maps/solution_2_boxes.pkl')
    #hyper-parameters settings
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--entropy_coef', type=float, default=0.1)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--rolloutStorage_size', type=int, default=5)
    #expert settings
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='scratch')

    args = parser.parse_args()

    for run in range(args.runs):
        
        wandb_session = wandb.init(project=args.env, config=vars(args), name="run-%i"%(run), reinit=True, group=args.mode)

        config = wandb.config
        train(config, wandb_session)
        wandb_session.finish()

if __name__ == "__main__":
    main()
