import gym
import gym_sokoban

from multiprocessing_env import SubprocVecEnv

def make_env():
    def _thunk():
        env = gym.make('Curriculum-Sokoban-v2', data_path = '../../maps/2_boxes/')
        return env
    return _thunk

envs = [make_env() for i in range(3)]
envs = SubprocVecEnv(envs)
import ipdb; ipdb.set_trace()
envs.reset()
