"""
Classic long double-pole balancing problem system,
see: Pagliuca P., Milano N. and Nolfi S. (2018). Maximizing adaptive power in neuroevolution. PLoS ONE 13(7): e0198788.
Variation of the classic cart-pole system implemented by Rich Sutton et al.
"""
import os
import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

# import time

import ErPredprey


class PredPreyEnv(gym.Env):
    def render(self, mode="human"):
        self.env.render()

    def __init__(self, render=False, n_robots=2):
        # action encodes left, right wheel and sound signal
        self.action_space = spaces.Box(-1., 1., shape=(6,), dtype='float32')
        # observation encodes various sensors
        self.observation = []
        self.observation_space = spaces.Box(-1, 1, shape=(36,), dtype='float32')
        # make the environment
        self.env = ErPredprey.PyErProblem()
        # create vector for observation, action, and done
        # and share the links with c++/cython library
        self.ob = np.arange(self.env.ninputs*n_robots, dtype=np.float32)
        self.ac = np.arange(self.env.noutputs*n_robots, dtype=np.float32)
        self.done = np.arange(1, dtype=np.intc)
        self.env.copyObs(self.ob)
        self.env.copyAct(self.ac)
        self.env.copyDone(self.done)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.env.reset()
        return self.ob

    def step(self, action):
        # copy action into the self.ac vector
        self.ac = action
        reward = self.env.step()
        return self.ob, reward, self.done, {}
