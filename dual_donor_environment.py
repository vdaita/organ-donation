from gymnasium import gym
import numpy as np
import networkx as nx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
import time

class DualDonorPairedKidneyDonationEnv(gym.Env):
    def __init__(
        self,
        n_agents: int = 100,
        n_timesteps: int = 64,
        departure_time_avg: int = 8,
        departure_time_std: float = 2
    ):
        super(DualDonorPairedKidneyDonationEnv, self).__init__()
        self.n_agents = n_agents
        self.n_timesteps = n_timesteps
        self.departure_time_avg = departure_time_avg
        self.departure_time_std = departure_time_std

        