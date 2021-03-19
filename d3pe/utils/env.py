import gym
import neorl

def get_env(task : str) -> gym.Env:
    if task in ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'ib', 'finance', 'citylearn']:
        env = neorl.make(task)
    else:
        env = gym.make(task)
    return env