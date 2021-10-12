import numpy as np
import torch

from experts.utils import get_distance

class Expert():
    '''
        need to deal with batch of states;
    '''
    def __init__(self, mode='search', pre_trained_path=None, expert_model=None, normalization=True):
        self.mode = mode
        if (mode == 'pre-train' and pre_trained_path == None) or (mode == 'pre-train' and expert_model == None):
            raise ValueError('You are using pre-train mode, but did not provide a pretrained path.')

        if mode == 'search':
            self.get_potentials = get_distance_for_all
        elif mode == 'pre-train':
            ckp = torch.load(pre_trained_path)
            expert_model.load_state_dict(ckp['a2c'])
            expert_model.cuda()
            expert_model.eval()
            self.get_potentials = expert_model.get_critic #we just need to use pre-trained model to forward the state in order to get q-values of states;
        elif mode == 'manhattan':
            self.get_potentials = get_potential_from_manhattan
        else:
            raise NotImplementedError('Other methods are not implemented yet.')


    def _mode(self):
        return self.mode

def get_distance_for_all(states):
    potentials = []
    for s in states:
        potential = -get_distance(s)
        potentials.append(potential)
    return torch.tensor(potentials).unsqueeze(1)

def get_potential_from_manhattan(state):
    pass
