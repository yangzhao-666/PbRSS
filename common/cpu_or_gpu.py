import torch
import torch.autograd as autograd

def cpu_or_gpu(USE_CUDA, actor_critic, rollout):
    if USE_CUDA:
        if not torch.cuda.is_available():
            raise ValueError('You wanna use cuda, but the machine you are on doesnt support')
        elif torch.cuda.is_available():
            Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
            actor_critic.cuda()
            rollout.cuda()
    else:
        Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
    
    return Variable, actor_critic, rollout
