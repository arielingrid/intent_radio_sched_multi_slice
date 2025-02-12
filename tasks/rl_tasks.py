from collections import deque
import cv2
import os
import gin
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import time
from os import getcwd
from tqdm import tqdm
from tasks.base_task import BaseTask
from agents.rl_simple import PIAgent
from agents.simu_rl_example import PIAgentSched
from associations.mult_slice import MultSliceAssociation
from channels.quadriga_seq import QuadrigaChannelSeq
from traffics.mult_slice import MultSliceTraffic
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import Agent, MARLCommEnv


seed = 10

class RLTask(BaseTask):
    """RL base task."""

    def __init__(self, v=True):
        self.env = None
        self.render = False
        self.step_cnt = 0
        self.eval_mode = False
        self.verbose = v

    def reset_for_rollout(self):
        self.step_cnt = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def modify_obs(self, obs):
        return obs

    def modify_action(self, act):
        return act

    def modify_reward(self, reward, done):
        return reward

    def modify_done(self, reward, done):
        return done

    def show_gui(self):
        if self.render and hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        self.env.close()

    def rollout(self, solution, evaluation=False):
        self.eval_mode = evaluation
        self.reset_for_rollout()
        solution.reset()
        if hasattr(self, 'register_solution'):
            self.register_solution(solution)

        start_time = time.time()
        #obs = self.env.reset(options={"initial_episode": self.testing_episodes})
        obs = self.env.reset()
        
        #obs = self.modify_obs(obs[0])
        ep_reward = 0
        done = False
        while not done:
            action = solution.get_action(obs)
            obs, reward, _, done, info = self.env.step(action)
            reward = self.modify_reward(reward, done)
            print("reward:", reward)
            self.step_cnt += 1
            ep_reward += reward
        ep_reward = np.mean(ep_reward)
        print("epi reward:", ep_reward)
        time_cost = time.time() - start_time
        if self.verbose:
            print('Rollout time={0:.2f}s, steps={1}, reward={2:.2f}'.format(
                time_cost, self.step_cnt, ep_reward))
        
        return ep_reward

@gin.configurable
class SchedulingEnvTask(RLTask):
    """Simple scheduling task toy example."""
        
    def __init__(
        self,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        seed: int = np.random.randint(1000),
        agent_name: str = "pi_agent",
        shuffle_on_reset=False,
        v=True,
        num_noise_channels=0
    ):  
        self.env = MARLCommEnv(
        QuadrigaChannelSeq,                       
        MultSliceTraffic,
        SimpleMobility,
        MultSliceAssociation,
        "mult_slice",
        )

        self.agent = PIAgentSched(
            self.env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs
        )
        self.env.set_agent_functions(
        self.agent.obs_space_format,
        self.agent.action_format,
        self.agent.calculate_reward,
        self.agent.get_obs_space(),
        self.agent.get_action_space(),
    )
        self.agent_name = agent_name
        self.shuffle_on_reset = shuffle_on_reset
        self.testing_episodes = 0
        self.verbose = v
        #self.obs_space = self.env.reset(seed)[0]
        #self.perm_ix = np.arange(self.obs_space.shape[0])
        #self.noise_std = 0.1
        #self.num_noise_channels = num_noise_channels
        self.rnd = np.random.RandomState(seed=0)

    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed=seed)
        return super(SchedulingEnvTask, self).seed(seed)

    """ def reset_for_rollout(self):
        self.shuffle_on_reset = False #Added to shuffle obs on reset
        #self.perm_ix = np.arange(self.env.observation_space.shape[0])
        self.perm_ix = np.arange(self.obs_space.shape[0])
        
        if self.shuffle_on_reset:
            self.rnd.shuffle(self.perm_ix)
            
        if self.verbose:
            print('perm_ix: {}'.format(self.perm_ix))
        return super(SchedulingEnvTask, self).reset_for_rollout()

    def modify_obs(self, obs):
        obs=obs[self.perm_ix]
        #Added next line to randomize the # of noise channels 
        #self.num_noise_channels = np.random.randint(2,11)
        if self.num_noise_channels > 0:
            noise_obs = self.rnd.randn(self.num_noise_channels) * self.noise_std
            obs = np.concatenate([obs, noise_obs], axis=0)
        return obs
 """
