import gymnasium as gym
import numpy as np
import rustworkx as rx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time
import copy
import numba as nb

class PrioritySelectionPairedKidneyDonationEnv(gym.Env):
    def __init__(self, n_agents=1000, p=0.037, q=0.087, pct_hard=0.6, arrival_rate=1, death_time=350, n_timesteps=700, seed=-1):
        self.n_agents = n_agents

        self.p = p
        self.q = q
        self.pct_hard = pct_hard
        self.arrival_rate = arrival_rate
        self.death_time = death_time
        self.n_timesteps = n_timesteps
        self.observation_space = Dict({
            "adjacency": MultiBinary((n_agents, n_agents)),
            "timestep": Discrete(n_timesteps),
            "arrivals": Box(low=0, high=n_timesteps, shape=(n_agents,), dtype=np.int32),
            "departures": Box(low=0, high=float('inf'), shape=(n_agents,), dtype=np.float32),
            "is_hard": MultiBinary(n_agents),
            "active": MultiBinary(n_agents),
            "matched": MultiBinary(n_agents),
            "total_timesteps": Discrete(n_timesteps + 1)
        })
        
        self.action_space = gym.spaces.MultiBinary(self.n_agents)
        self.seed = seed
        if seed >= 0:
            self.reset(seed=seed)
        else:
            self.reset()

        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if not seed:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed
        super().reset(seed=seed)
        
        start_time = time.time()
        # Reset compatibility matrix
        self.compatibility = np.zeros((self.n_agents, self.n_agents))
        
        # Identify agents that are hard to match (e.g., highly sensitized patients)
        hard_indices = self.np_random.choice(self.n_agents, int(self.n_agents * self.pct_hard), replace=False)
        self.is_hard_to_match = np.zeros(self.n_agents)
        self.is_hard_to_match[hard_indices] = 1
        # print("Number of hard indices: ", len(hard_indices), "Total agents: ", self.n_agents, "Hard to match proportion: ", sum(self.is_hard_to_match) / self.n_agents)

        easy_indices = np.setdiff1d(np.arange(self.n_agents), hard_indices)

        random_values = self.np_random.random((self.n_agents, self.n_agents))
        random_values = np.triu(random_values, k=1) + np.triu(random_values, k=1).T  # Make it symmetric

        # Fill compatibility matrix based on hard/easy matching criteria
        self.compatibility[np.ix_(hard_indices, easy_indices)] = random_values[np.ix_(hard_indices, easy_indices)] < self.p  # Hard-to-Easy
        self.compatibility[np.ix_(easy_indices, hard_indices)] = random_values[np.ix_(easy_indices, hard_indices)] < self.p  # Easy-to-Hard
        self.compatibility[np.ix_(easy_indices, easy_indices)] = random_values[np.ix_(easy_indices, easy_indices)] < self.q  # Easy-to-Easy
        self.compatibility[np.ix_(hard_indices, hard_indices)] = 0  # Hard-to-Hard
        self.compatibility[np.arange(self.n_agents), np.arange(self.n_agents)] = 0 # no self-compatibility

        # Generate arrivals using exponential distribution
        self.arrival_times = np.linspace(0, self.n_timesteps - self.death_time, self.n_agents).astype(int)
        
        # Generate departure times based on arrival times plus criticality duration
        exponential_distributions = self.np_random.exponential(1, (self.n_agents, )) * self.death_time
        exponential_distributions = exponential_distributions.astype(int)
        self.real_departure_times = np.minimum(self.arrival_times + exponential_distributions, np.ones(self.n_agents) * self.n_timesteps)
        
        self.active_agents = np.zeros(self.n_agents)
        self.matched_agents = np.zeros(self.n_agents)
        self.time_matched = np.ones(self.n_agents) * -1
        end_time = time.time()

        self.current_step = 1 # fix to make sure that we have the first few elements
        self.current_graph = rx.PyDiGraph()
        for i in range(self.n_agents):
            self.current_graph.add_node(i)
        return self.get_observation(), self.get_info()

    def start_over(self): # start over in a new environment with a similar setup
        return self.reset(seed=self.seed)

    def get_observation(self):
        return {
            "adjacency": rx.adjacency_matrix(self.current_graph).astype(np.int8) if self.current_graph else np.zeros((self.n_agents, self.n_agents)),
            "timestep": self.current_step,
            "arrivals": self.arrival_times.astype(np.int8),
            "departures": self.real_departure_times.astype(np.int8),
            "is_hard": self.is_hard_to_match.astype(np.int8),
            "active": self.active_agents.astype(np.int8),
            "matched": self.matched_agents.astype(np.int8),
            "total_timesteps": self.n_timesteps
        }
    
    def clear_node_edges(self, node):
        in_edges = list(self.current_graph.in_edges(node))
        out_edges = list(self.current_graph.out_edges(node))

        for u, v, w in in_edges + out_edges:
            if self.current_graph.has_edge(u, v):
                self.current_graph.remove_edge(u, v)

    def node_matched(self, u):
        self.clear_node_edges(u)
        self.active_agents[u] = 0
        self.matched_agents[u] = 1
        self.time_matched[u] = self.current_step

    def match_subgraph(self, selected_nodes, matchable_nodes, adj_matrix):        
        graph = rx.PyGraph()
        graph.add_nodes_from(range(self.n_agents))
        rx_edges = get_edges(selected_nodes, matchable_nodes, adj_matrix)
        graph.add_edges_from(rx_edges)

        # Find the maximum matching
        best_matching = rx.max_weight_matching(graph, max_cardinality=True)
        for u, v in best_matching: # we already know they are getting matched
            self.node_matched(u)
            self.node_matched(v)

    def step(self, action, **kwargs):
        previous_matched = np.sum(self.matched_agents)

        selected_nodes = np.where(action == 1)[0]
        hard_indices = np.where(self.is_hard_to_match == 1)[0]
        
        adj_matrix = rx.adjacency_matrix(self.current_graph)
        
        self.match_subgraph(selected_nodes, hard_indices, adj_matrix)
        self.match_subgraph(selected_nodes, np.arange(self.n_agents), adj_matrix)

        self.manage_arrivals_departures()
        self.current_step += 1
        done = self.current_step >= self.n_timesteps
        
        current_matched = np.sum(self.matched_agents)
        reward = (current_matched - previous_matched) / self.n_agents

        return self.get_observation(), reward, done, {}, {}

    def manage_arrivals_departures(self):
        # add the new arrivals to the graph 
        new_arrivals = np.where(self.arrival_times == self.current_step)[0]

        for agent_idx in new_arrivals:
            self.active_agents[agent_idx] = 1
            # Only connect to other ACTIVE agents
            for other_agent in self.current_graph.nodes():
                if other_agent != agent_idx and self.active_agents[other_agent] == 1:
                    if self.compatibility[agent_idx, other_agent] == 1:
                        self.current_graph.add_edge(agent_idx, other_agent, 1)
                    if self.compatibility[other_agent, agent_idx] == 1:
                        self.current_graph.add_edge(other_agent, agent_idx, 1)

        # Clear edges for departures
        departures = np.where(self.real_departure_times == self.current_step)[0]
        # print("Departures: ", departures, " which are leaving after: ", self.real_departure_times[departures] - self.arrival_times[departures])
        for agent_idx in departures:
            if self.active_agents[agent_idx] == 1:  # Only process active agents
                self.clear_node_edges(agent_idx)
                self.active_agents[agent_idx] = 0

    def get_info(self):
        arrived_mask = self.arrival_times <= self.current_step
        hard_mask = self.is_hard_to_match == 1
        
        hard_arrived = np.logical_and(hard_mask, arrived_mask)
        regular_arrived = np.logical_and(~hard_mask, arrived_mask)
        
        return {
            "hard_percentage": np.mean(self.is_hard_to_match),
            "hard_to_match_rate": np.mean(self.matched_agents[hard_arrived]),
            "regular_match_rate": np.mean(self.matched_agents[regular_arrived]),
            "active_agents": np.sum(self.active_agents),
            "total_matched": np.sum(self.matched_agents),
            "current_step": self.current_step,
            "current_progress_percentage": self.current_step / self.n_timesteps
        }

    def render(self):
        plt.imshow(self.get_observation()["adjacency_matrix"])
        plt.show()

    def get_greedy_percentage(self):
        obs, _ = self.start_over()
        done = False
        while not done:
            action = np.ones(self.n_agents)
            obs, reward, done, _, _ = self.step(action)
        total_reward = np.sum(self.matched_agents) / self.n_agents

        hard_waiting, easy_waiting = self.get_waiting_time()

        return total_reward, hard_waiting, easy_waiting

    def get_patient_percentage(self):
        obs, _ = self.start_over()
        done = False
        while not done:
            action = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                if self.real_departure_times[i] - self.current_step == 1:
                    action[i] = 1
            obs, reward, done, _, _ = self.step(action)
        total_reward = np.sum(self.matched_agents) / self.n_agents

        return total_reward, self.get_hard_waiting_time(), self.get_easy_waiting_time()
    
    def get_waiting_time(self, is_hard):
        difficulty = 1 if is_hard else 0
        waiting_times = self.time_matched[self.is_hard_to_match == difficulty] - self.arrival_times[self.is_hard_to_match == difficulty]
        waiting_times = waiting_times[waiting_times > 0]
        return waiting_times
    
    def get_hard_waiting_time(self):
        return self.get_waiting_time(is_hard=True)
    
    def get_easy_waiting_time(self):
        return self.get_waiting_time(is_hard=False)
    
    def calculate_theoretical_max(self):
        # based on compat and arrival/departure times, compute a theoretical maximum
        graph = rx.PyGraph()
        for i in range(self.n_agents):
            graph.add_node(i)
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if self.compatibility[i, j] == 1 and self.compatibility[j, i] == 1:
                    range_i = [self.arrival_times[i], self.real_departure_times[i]]
                    range_j = [self.arrival_times[j], self.real_departure_times[j]]
                    # if overlap, then add the edge
                    if range_i[0] < range_j[1] and range_j[0] < range_i[1]:
                        graph.add_edge(i, j, 1)
        
        # find the maximum matching
        matching = rx.max_weight_matching(graph, max_cardinality=True)
        return 2 * len(matching) / self.n_agents
    
@nb.njit(fastmath=True, parallel=True)
def get_edges_cycles(selected_nodes, matchable_nodes, adj_matrix):
    adj_view = adj_matrix[selected_nodes][:, matchable_nodes]
    edges = np.argwhere(adj_view == 1)
    edges = [(selected_nodes[edge[0]], matchable_nodes[edge[1]], 1) for edge in edges]
    return edges

@nb.njit(fastmath=True, parallel=True)
def get_edges(selected_nodes, matchable_nodes, is_hard_to_match, adj_matrix):
    adj_bidirectional = np.logical_and(adj_matrix, adj_matrix.T)
    adj_view = adj_bidirectional[selected_nodes][:, matchable_nodes]
    edges = np.argwhere(adj_view == 1)
    
    edges = []
    for edge in edges:
        a = selected_nodes[edge[0]]
        b = matchable_nodes[edge[1]]

        if is_hard_to_match[a] == 1 or is_hard_to_match[b] == 1:
            edges.append((a, b, 1 + 1e-6))
        else:
            edges.append((a, b, 1))
    return edges