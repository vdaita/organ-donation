"""
Currently, the environment is based on using a GNN-based model to classify, for each node, whether or not it should be selected to be put into a greedy matching scheme. However, it could be more efficient to generate cycles using a greedy algorithm or some other system, and then classify whether or those those embedded cycles should be selected right now or later.

Currently, for the purposes of simplifying, I'm using pairs only.
"""

import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Optional
from gymnasium.spaces import Discrete, Box, Dict, MultiBinary


def intersect_ranges(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])

class BinaryDecisionEnvironment(gym.Env):
    def __init__(
        self,
        n_agents: int = 100,
        n_timesteps: int = 64,
        departure_time_avg: int = 8,
        departure_time_std: float = 2,
        easy_to_easy_rate: float = 0.087,
        easy_to_hard_rate: float = 0.037,
        percent_hard: float = 0.7
    ):
        super(BinaryDecisionEnvironment, self).__init__()
        self.n_agents = n_agents
        self.n_timesteps = n_timesteps
        self.departure_time_avg = departure_time_avg
        self.departure_time_std = departure_time_std
        self.easy_to_easy_rate = easy_to_easy_rate
        self.easy_to_hard_rate = easy_to_hard_rate
        self.percent_hard = percent_hard

        self.action_space = MultiBinary(1)
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(15, ),
            dtype=np.float32
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        if not seed:
            seed = np.random.randint(0, 2 ** 32 - 1)
        self.seed = seed
        super().reset(seed=seed)

        self.compat = np.zeros((self.n_agents, self.n_agents))
        self.graph = nx.DiGraph()
        self.current_timestep = 0

        random_values = self.np_random.random((self.n_agents, self.n_agents))

        self.arrival_times = self.np_random.integers(0, self.n_timesteps, size=self.n_agents)
        durations = self.np_random.normal(
            self.departure_time_avg,
            self.departure_time_std,
            size=self.n_agents
        )
        durations = np.maximum(durations, 1)
        self.departure_times = np.minimum(
            self.n_timesteps,
            self.arrival_times + durations
        ).astype(int)

        self.is_hard = np.zeros(self.n_agents)
        hard_count = int(self.n_agents * self.percent_hard)
        hard_idxs = self.np_random.choice(self.n_agents, size=hard_count, replace=False)
        easy_idxs = np.array([i for i in range(self.n_agents)if i not in hard_idxs])
        self.is_hard[hard_idxs] = 1

        self.compat[np.ix_(easy_idxs, easy_idxs)] = random_values[np.ix_(easy_idxs, easy_idxs)] < self.easy_to_easy_rate

        self.compat[np.ix_(easy_idxs, hard_idxs)] = random_values[np.ix_(easy_idxs, hard_idxs)] < self.easy_to_hard_rate

        self.compat[np.ix_(hard_idxs, easy_idxs)] = random_values[np.ix_(hard_idxs, easy_idxs)] < self.easy_to_hard_rate

        self.compat[np.ix_(hard_idxs, hard_idxs)] = 0 
        self.compat[np.arange(self.n_agents), np.arange(self.n_agents)] = 0


        self.active_agents = np.zeros(self.n_agents)
        self.matched_agents = np.zeros(self.n_agents)
        
        self.vetoed_cycles = np.zeros((self.n_agents, self.n_agents))

        return self._get_obs(), {}

    def find_edge(self):
        htm_edges = []
        etm_edges = []

        all_edges = list(self.graph.edges())
        all_edges = sorted(all_edges, key=lambda x: (x[0], x[1]))

        for edge in all_edges:
            a, b = edge
            if self.is_hard[a] == 1 and self.is_hard[b] == 1 and not self.vetoed_cycles[a, b]:
                htm_edges.append(edge)
            elif not self.vetoed_cycles[a, b]:
                etm_edges.append(edge)
        if htm_edges:
            return htm_edges[0]
        elif etm_edges:
            return etm_edges[0]
        else:
            return None
        
    def add_node(self, node):
        self.graph.add_node(node)
        self.active_agents[node] = 1
        for i in range(self.n_agents):
            if self.compat[node, i] == 1 and self.active_agents[i] == 1:
                self.graph.add_edge(node, i)
                self.graph.add_edge(i, node)
    
    def remove_node(self, node):
        self.graph.remove_node(node)
        self.active_agents[node] = 0

    def step(self, action):
        if action:
            for i in self._get_edge():
                self.matched_agents[i] = 1
                self.active_agents[i] = 0
                self.graph.remove_node(i)
        else:
            a, b = self._get_edge()
            self.vetoed_cycles[a, b] = 1
            self.vetoed_cycles[b, a] = 1

        # find a different edge
        new_obs = self._get_obs()
        # print("Observation: ", new_obs)
        done = False

        if self.current_timestep >= self.n_timesteps:
            done = True
            reward = self._get_reward()
            return new_obs, reward, done, done, {}
        
        return new_obs, 0, done, done, {}

    def _edge_to_feature(self, edge, current_timestep=None):
        if current_timestep is None:
            current_timestep = self.current_timestep
        a, b = edge
        features = np.zeros(15,)
        # print("Edge: ", edge, "Current timestep: ", self.current_timestep, "Departure time a: ", self.departure_times[a], "Departure time b: ", self.departure_times[b], "Arrival time a: ", self.arrival_times[a], "Arrival time b: ", self.arrival_times[b])
        features[0] = current_timestep / self.n_timesteps
        features[1] = (self.departure_times[a] - current_timestep) / (self.departure_times[a] - self.arrival_times[a])
        features[2] = (self.departure_times[b] - current_timestep) / (self.departure_times[b] - self.arrival_times[b])
        features[3] = self.is_hard[a]
        features[4] = self.is_hard[b]
        features[5] = current_timestep % (int(self.n_timesteps / 16))
        features[6] = current_timestep % (int(self.n_timesteps / 8))
        features[7] = current_timestep % (int(self.n_timesteps / 4))
        features[8] = min(self.departure_times[a] - current_timestep, self. departure_times[b] - current_timestep) / self.n_timesteps

        features[9] = 1.0 if (self.departure_times[a] - current_timestep) <= 2 else 0.0 
        features[10] = 1.0 if (self.departure_times[b] - current_timestep) <= 2 else 0.0 
        
        # number of easy edges for a
        features[11] = np.sum(self.compat[a, :] * self.active_agents * (1 - self.is_hard)) / np.sum(self.active_agents)
        # number of easy edges for b
        features[12] = np.sum(self.compat[b, :] * self.active_agents * (1 - self.is_hard)) / np.sum(self.active_agents)

        # number of hard edges for a
        features[13] = np.sum(self.compat[a, :] * self.active_agents * self.is_hard) / np.sum(self.active_agents)
        # number of hard edges for b
        features[14] = np.sum(self.compat[b, :] * self.active_agents * self.is_hard) / np.sum(self.active_agents)

        return features
    
    def _get_reward(self):
        return sum(self.matched_agents) / self.n_agents
    
    def _get_edge(self):
        edge = self.find_edge()
        while not edge:
            self.current_timestep += 1 # there is no edge in the current timestep
            self.vetoed_cycles = np.zeros((self.n_agents, self.n_agents)) # reset the vetoed cycles - should be able to choose at this new timestep, with new circumstances

            if self.current_timestep > self.n_timesteps:
                return [0, 0]
            # add arrivals
            new_arrivals = np.where(self.arrival_times == self.current_timestep)[0]
            for i in new_arrivals:
                self.add_node(i)
            # remove departures
            new_departures = np.where(self.departure_times == self.current_timestep)[0]
            for i in new_departures:
                if self.active_agents[i] == 1:
                    self.remove_node(i)

            edge = self.find_edge()
        return edge
    
    def _get_obs(self): # only returns None when the simulation is over
        edge = self._get_edge()
        # print("Edge: ", edge)
        return self._edge_to_feature(edge)
    
    def get_greedy_result(self):
        obs, _ = self.reset(seed=self.seed)
        done = False
        reward = 0
        while not done:
            action = 1 
            obs, reward, done, _, _ = self.step(action)
        return reward
    
    def get_theoretical_max(self): # 
        tm_graph = nx.Graph()
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                
                if self.compat[i, j] == 1:
                    i_times = [self.arrival_times[i], self.departure_times[i]]
                    j_times = [self.arrival_times[j], self.departure_times[j]]
                    if intersect_ranges(i_times, j_times):
                        tm_graph.add_edge(i, j)

        best_matching = nx.max_weight_matching(tm_graph, maxcardinality=True)
        best_matching = list(best_matching)
        all_edges = list(tm_graph.edges())
        return best_matching, all_edges
                    
 