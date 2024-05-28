import math

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding
from scipy.linalg import expm


class Linearized_Simple_Pendulum(gym.Env):
    def __init__(self, cfg):
        self.g = cfg.g
        self.m = cfg.m
        self.l = cfg.l
        self.b = cfg.b
        self.action_scale = cfg.action_scale
        self.action_lim = cfg.action_lim
        self.theta_lim = cfg.theta_lim
        self.theta_dot_lim = cfg.theta_dot_lim
        self.delta_t = cfg.delta_t
        self.evaluate = cfg.evaluate
        self.reward_scale = cfg.reward_scale

        high = np.array(
            [
                self.theta_lim,
                self.theta_dot_lim
            ],
            dtype=np.float32,
        )

        self.A = np.array([[0, 1],
                           [self.g / self.l, -self.b / (self.m * self.l ** 2)]], dtype=np.float32)

        self.B = np.array([[0],
                           [1 / (self.m * self.l ** 2)]])

        self.G = expm(self.A * self.delta_t)
        self.H = np.linalg.inv(self.A) @ (self.G - np.eye(2)) @ self.B

        self.Q = np.array(cfg.Q).reshape((2, 2))
        self.R = cfg.R

        self.process_noise_cov = np.array(cfg.process_noise_cov).reshape((2, 2))
        self.measure_noise_cov = np.array(cfg.measure_noise_cov).reshape((2, 2))

        act_high = np.array([self.action_lim, ], dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.action = np.clip(action*self.action_scale, -self.action_lim, self.action_lim)
        self.state = self.G @ self.state + self.H * self.action_scale * action.squeeze() + np.random.multivariate_normal(
            np.array([0, 0]), self.process_noise_cov)[:, None]
        reward = self.reward_fcn(self.state, action)
        done = self.termination_fcn(self.state, action)

        return self.state.squeeze() + np.random.multivariate_normal(np.array([0, 0]),
                                                                    self.measure_noise_cov), reward.squeeze(), done, {}

    def reset(self):
        if not self.evaluate:
            theta = self.np_random.uniform(low=-self.theta_lim, high=self.theta_lim, size=(1,))
            theta_dot = self.np_random.uniform(low=-self.theta_dot_lim, high=self.theta_dot_lim, size=(1,))
        else:
            theta = np.random.choice(np.array([-0.7854, 0.7854]), size=1)
            theta_dot = np.array([0])
        self.state = np.array([[theta[0], theta_dot[0]]]).T
        if self.evaluate: print("eval episode started with state", self.state)
        # self.state = np.array([[np.pi/4 - 1e-3, 0]]).T
        # if self.action_grad:
        #     self.action = 0
        return self.state.squeeze()

    def termination_fcn(self, state, action):
        if np.abs(state[0, :]) > self.theta_lim or \
                np.abs(state[1, :]) > self.theta_dot_lim:
            if self.evaluate: print("eval episode terminated with state", state)
            return True
        else:
            return False

    def reward_fcn(self, state, action):
        return self.reward_scale * ((self.theta_lim ** 2 * self.Q[0, 0] + self.theta_dot_lim ** 2 * self.Q[
            1, 1] + self.action_lim ** 2 * self.R) - (state.T @ self.Q @ state + self.R * action ** 2)) / (
                           self.theta_lim ** 2 * self.Q[0, 0] + self.theta_dot_lim ** 2 * self.Q[
                       1, 1] + self.action_lim ** 2 * self.R)
        # return -(state.T @ self.Q @ state + self.R * action ** 2)
        # return self.reward_scale * ((self.theta_lim ** 2 * self.Q[0,0] + self.theta_dot_lim ** 2 * self.Q[1,1] + self.action_lim ** 2 * self.R) -(state.T @ self.Q @ state + self.R * action ** 2))
