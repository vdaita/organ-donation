import gymnasium as gym
import numpy as np
import rustworkx as rx
from typing import Optional
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box

def do_ranges_overlap(range1, range2):
    return not (range1[1] < range2[0] or range2[1] < range1[0])

class AdvanceKnowledgeSimulationEnvironment(gym.Env):
    def __init__(
        self,
        n_agents: int = 100,
        n_timesteps: int = 64,
        death_time: int = 32,
        p: float = 0.037,
        q: float = 0.087,
        pct_hard: float = 0.6,
        warning_mean: int = 8, # when you know that someone is probably going to join the kidney exchange
        seed: Optional[int] = None
    ):
        super(AdvanceKnowledgeSimulationEnvironment, self).__init__()
        self.n_agents = n_agents
        self.n_timesteps = n_timesteps
        self.death_time = death_time
        self.p = p
        self.q = q
        self.pct_hard = pct_hard
        self.warning_mean = warning_mean

        self.action_space = Discrete(2) 
        self.observation_space = ...

        if seed >= 0:
            self.reset(seed=seed)
        else:
            self.reset()
    
    def reset(self, seed: Optional[int] = None):
        if not seed:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed
        super().reset(seed=seed)

        arrival_rate = self.n_agents / self.n_timesteps
        self.arrivals = self.np_random.poisson(arrival_rate, size=self.n_agents)
        
        knowledge_time = self.np_random.exponential(self.warning_mean, size=self.n_agents)
        self.knowledge = np.clip(self.arrivals - knowledge_time, 0, self.n_timesteps - 1)

        time_in_exchange = np.random.exponential(self.death_time, size=self.n_agents)
        self.departures = np.clip(self.arrivals + time_in_exchange, 0, self.n_timesteps - 1)

        hard_indices = self.np_random.choice(self.n_agents, int(self.n_agents * self.pct_hard), replace=False)
        self.is_hard_to_match = np.zeros(self.n_agents)
        self.is_hard_to_match[hard_indices] = 1

        easy_indices = np.setdiff1d(np.arange(self.n_agents), hard_indices)

        self.compat = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        random_matrix = self.np_random.random((self.n_agents, self.n_agents))
        
        self.compat[easy_indices, hard_indices] = random_matrix[easy_indices, hard_indices] < self.p
        self.compat[hard_indices, easy_indices] = random_matrix[hard_indices, easy_indices] < self.p
        self.compat[easy_indices, easy_indices] = random_matrix[easy_indices, easy_indices] < self.q
        self.compat[hard_indices, hard_indices] = 0
        self.compat[np.arange(self.n_agents), np.arange(self.n_agents)] = 0

        self.waiting_agents = np.zeros(self.n_agents)
        self.active_agents = np.zeros(self.n_agents)
        self.matched_agents = np.zeros(self.n_agents)
        self.time_matched = np.ones(self.n_agents) * -1

        self.current_step = 0

        # waiting graph: includes nodes that are going to arrive
        self.waiting_graph = rx.PyGraph() 
        # current graph: includes nodes that are already in the exchange
        self.current_graph = rx.PyGraph()

        self.waiting_graph.add_nodes_from(range(self.n_agents))
        self.current_graph.add_nodes_from(range(self.n_agents))

        return self._get_observation(), {}

    def step(self, action): # 1 if using the waiting graph, 0 if using the current graph
        # match the agent according to the current graph
        if action == 1:
            graph = self.waiting_graph
        else:
            graph = self.current_graph
        
        # find matching within the graph
        matching = rx.max_weight_matching(graph, max_cardinality=True)
        for node1, node2 in matching:
            arrival_1 = self.arrivals[node1]
            departure_1 = self.departures[node1]

            arrival_2 = self.arrivals[node2]
            departure_2 = self.departures[node2]

            if self.current_step >= arrival_1 and self.current_step <= departure_1:
                if self.current_step >= arrival_2 and self.current_step <= departure_2:
                    self.matched_agents[node1] = 1
                    self.matched_agents[node2] = 1
                    self.active_agents[node1] = 0
                    self.active_agents[node2] = 0
                    self.time_matched[node1] = self.current_step
                    self.time_matched[node2] = self.current_step

                    self.waiting_graph.remove_node(node1)
                    self.current_graph.remove_node(node1)

        # check arrivals to the waiting graph
        waiting_arrivals = np.where(self.knowledge == self.current_step)[0]
        for waiting_arrival_node in waiting_arrivals:
            compatible_nodes = np.where(self.compat[waiting_arrival_node] == 1)[0]
            for compat_node in compatible_nodes:
                if self.compat[compat_node, waiting_arrival_node] == 1:
                    waiting_node_range = [self.arrivals[waiting_arrival_node], self.departures[waiting_arrival_node]]
                    compat_node_range = [self.arrivals[compat_node], self.departures[compat_node]]
                    if do_ranges_overlap(waiting_node_range, compat_node_range):
                        self.waiting_graph.add_edge(waiting_arrival_node, compat_node)

        # check arrivals to the current graph
        arrivals = np.where(self.arrivals == self.current_step)[0] 
        for arrival_node in arrivals:
            compatible_nodes = np.where(self.compat[arrival_node] == 1)[0]
            for compat_node in compatible_nodes:
                if self.compat[compat_node, arrival_node] == 1:
                    self.current_graph.add_edge(arrival_node, compat_node)
        self.active_agents[arrivals] = 1

        # checking departures from the waiting and current graphs
        departures = np.where(self.departures == self.current_step)[0] 
        self.waiting_graph.remove_nodes_from(departures)
        self.current_graph.remove_nodes_from(departures)
        self.active_agents[departures] = 0

        self.current_step += 1

        return self._get_observation(), self._get_reward(), self._is_done(), {}
    
    def _is_done(self):
        return self.current_step >= self.n_timesteps

    def _get_observation(self):
        return {
            "waiting_graph": self.waiting_graph,
            "current_graph": self.current_graph,
            "active_agents": self.active_agents,
            "matched_agents": self.matched_agents,
            "time_matched": self.time_matched
        }
    
    def _get_reward(self):
        return np.sum(self.matched_agents) / self.n_agents
    
    def compute_reward(self, use_waiting_graph: bool = True):
        obs, _ = self.reset(seed=self.seed)
        done = False
        while not done:
            action = 1 if use_waiting_graph else 0
            obs, reward, done, _ = self.step(action)
        return reward
    
