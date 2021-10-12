import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class OnPolicy(nn.Module):
    def __init__(self):
        super(OnPolicy, self).__init__()
        self.steps_done = 0

    def forward(self, x):
        raise NotImplementedError

    def select_action(self, x, test = 0, determinisitc=False):
        if(test == 1):
            with torch.no_grad():
                logit, value = self.forward(x)
        else:
            self.steps_done += 1
            logit, value = self.forward(x)

        probs = F.softmax(logit, dim=1)

        if determinisitc:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(num_samples=1)

        return action

    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)
        
        probs = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()

        return logit, action_log_probs, value, entropy
