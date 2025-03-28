import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
import matplotlib.pyplot as plt
from typing import Tuple


def generate_profile(n_agents) -> Tuple[Tuple, Tuple]:
    blood_types = ['']

def does_match(patient) -> bool:
    ...

class PairedKidneyDonationEnv(gym.Env):
    def __init__(self, n_agents=1000, p=0.037, q=0.087, pct_hard=0.34, arrival_rate=1, criticality_rate=400, n_timesteps=700):
        self.n_agents = n_agents

        self.p = p
        self.q = q
        self.pct_hard = pct_hard

        self.arrival_rate = arrival_rate # agent arrives to market at time t (t dist. exponentially with mean a)
        self.criticality_rate = criticality_rate # agent arrives to market at time t, becomes critical after Z units of time (Z dist. exponentially with mean d)
        
        self.n_timesteps = n_timesteps

        self.observation_space = Dict({
            "adjacency_matrix": MultiBinary((n_agents, n_agents)),
            "timestep": Discrete(n_timesteps)
        })

        self.action_space = Dict({
            "selection": MultiBinary(n_agents), # which agents should be matched selectively
            "match_selection": Discrete(2, start=0), # whether or not the selectively matched points should be matched
            "match_regular": Discrete(2, start=0) # whether or not the regular points should be matched at this timestep
        })

        self.compatibility = np.zeros((n_agents, n_agents))
        self.arrival_times = np.zeros(n_agents)
        self.real_departure_times = np.zeros(n_agents) # currently, we assume that each individual leaves the market after the same amount of time (w/ distribution)
        self.active_agents = np.zeros(n_agents) # 1 if the agent is in the market, 0 otherwise

        self.current_step = 0
        self.current_graph = nx.DiGraph()

        self.matched_pairs = 0

        
    def reset(self):
        # Reset compatibility matrix
        self.compatibility = np.zeros((self.n_agents, self.n_agents))
        
        # Identify agents that are hard to match (e.g., highly sensitized patients)
        hard_to_match = np.random.choice(self.n_agents, int(self.n_agents * self.pct_hard), replace=False)
        
        # Generate compatibility edges
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:  # No self-compatibility
                    if i in hard_to_match and j in hard_to_match:
                        # H-H pairs are never compatible
                        self.compatibility[i, j] = 0
                    elif i in hard_to_match or j in hard_to_match:
                        # E-H pairs are compatible with probability p
                        if np.random.rand() < self.p:
                            self.compatibility[i, j] = 1
                    else:
                        # E-E pairs are compatible with probability q
                        if np.random.rand() < self.q:
                            self.compatibility[i, j] = 1

        # Generate arrivals using exponential distribution
        self.arrival_times = np.zeros(self.n_agents, dtype=int)
        inter_arrival_times = np.random.exponential(1.0/self.arrival_rate, self.n_agents)
        cumulative_times = np.cumsum(inter_arrival_times)
        scaled_times = (cumulative_times / np.max(cumulative_times) * (self.n_timesteps * 0.9)).astype(int)
        self.arrival_times = np.clip(scaled_times, 0, self.n_timesteps-1)
        
        # Generate departure times based on arrival times plus criticality duration
        criticality_durations = np.random.exponential(self.criticality_rate, self.n_agents)
        self.real_departure_times = np.minimum((self.arrival_times + criticality_durations).astype(int), self.n_timesteps - 1)

        self.active_agents = np.zeros(self.n_agents)
        self.is_hard_to_match = np.zeros(self.n_agents)
        self.is_hard_to_match[hard_to_match] = 1

        print(f"Average arrival time: {np.mean(self.arrival_times):.2f}")
        print(f"Average criticality duration: {np.mean(criticality_durations):.2f}")

        print("Arrival times: ", self.arrival_times)
        print("Departure times: ", self.real_departure_times)

        self.current_step = 0
        self.current_graph = nx.DiGraph()
        self.matched_pairs = 0

        return self.get_observation()

    def get_observation(self):
        adj_matrix = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if self.compatibility[i, j] == 1:
                    if self.current_graph.has_node(i) and self.current_graph.has_node(j):
                        adj_matrix[i, j] = 1

        return {
            "adjacency_matrix": adj_matrix,
            "timestep": self.current_step
        }
    
    def get_reward(self):
        return self.matched_pairs * 2 / self.n_agents

    def step(self, action):
        # check if the priority nodes should be matched
        if action["match_selection"] == 1:
            priority_nodes = np.where(action["selection"] == 1)[0]
            # Only proceed if there are selected priority nodes and they exist in the graph
            if len(priority_nodes) > 0:
                # Create a subgraph with only the selected priority nodes and their neighbors
                selected_subgraph = nx.DiGraph()
                for node in priority_nodes:
                    if self.current_graph.has_node(node):  # Check if node exists in graph
                        selected_subgraph.add_node(node)
                        for neighbor in self.current_graph.neighbors(node):
                            selected_subgraph.add_edge(node, neighbor)
                            # Add the reverse edge if it exists (for bidirectional compatibility)
                            if self.current_graph.has_edge(neighbor, node):
                                selected_subgraph.add_edge(neighbor, node)
                
                # Convert to undirected graph for matching
                undirected_subgraph = nx.Graph()
                for u, v in selected_subgraph.edges():
                    # Only add edge if there's mutual compatibility
                    if selected_subgraph.has_edge(v, u):
                        undirected_subgraph.add_edge(u, v)
                
                # Find the best matching in this graph
                best_priority_matching = nx.max_weight_matching(undirected_subgraph, maxcardinality=True)
                self.matched_pairs += len(best_priority_matching)  # Changed from self.pairs to self.matched_pairs
                
                # Remove the matched nodes from the graph
                for u, v in best_priority_matching:
                    if self.current_graph.has_node(u):  # Check before removing
                        self.current_graph.remove_node(u)
                        self.active_agents[u] = 0
                    if self.current_graph.has_node(v):  # Check before removing
                        self.current_graph.remove_node(v)
                        self.active_agents[v] = 0

        if action["match_regular"] == 1:
            # Convert to undirected graph for matching
            undirected_graph = nx.Graph()
            for u, v in self.current_graph.edges():
                # Only add edge if there's mutual compatibility
                if self.current_graph.has_edge(v, u):
                    undirected_graph.add_edge(u, v)
            
            best_regular_matching = nx.max_weight_matching(undirected_graph, maxcardinality=True)
            self.matched_pairs += len(best_regular_matching)  # Changed from self.pairs to self.matched_pairs
            
            # Remove the matched nodes from the graph
            for u, v in best_regular_matching:
                if self.current_graph.has_node(u):  # Check before removing
                    self.current_graph.remove_node(u)
                    self.active_agents[u] = 0
                if self.current_graph.has_node(v):  # Check before removing
                    self.current_graph.remove_node(v)
                    self.active_agents[v] = 0

        # add the new arrivals to the graph 
        new_arrivals = np.where(self.arrival_times == self.current_step)[0]

        for agent_idx in new_arrivals:
            self.current_graph.add_node(agent_idx)
            self.active_agents[agent_idx] = 1
            for other_agent in self.current_graph.nodes():
                if other_agent != agent_idx:
                    if self.compatibility[agent_idx, other_agent] == 1:
                        self.current_graph.add_edge(agent_idx, other_agent)
                    if self.compatibility[other_agent, agent_idx] == 1:
                        self.current_graph.add_edge(other_agent, agent_idx)

        # remove the old departures
        departures = np.where(self.real_departure_times == self.current_step)[0]
        for agent_idx in departures:
            if agent_idx in self.current_graph.nodes():
                self.current_graph.remove_node(agent_idx)
                self.active_agents[agent_idx] = 0

        self.current_step += 1

        return self.get_observation(), self.get_reward(), self.current_step == self.n_timesteps, self.get_info()

    def get_info(self):
        num_hard_matched = sum([1 for i in range(self.n_agents) 
                            if self.is_hard_to_match[i] == 1 and self.active_agents[i] == 0 
                            and self.arrival_times[i] <= self.current_step])
        num_hard_total = sum(self.is_hard_to_match)
        
        num_regular_matched = self.matched_pairs * 2 - num_hard_matched
        num_regular_total = self.n_agents - num_hard_total
        
        return {
            "hard_to_match_rate": num_hard_matched / max(1, num_hard_total),
            "regular_match_rate": num_regular_matched / max(1, num_regular_total),
            "active_agents": sum(self.active_agents),
            "total_matched": self.matched_pairs * 2
        }
        
    def render(self):
        plt.imshow(self.get_observation()["adjacency_matrix"])
        plt.show()

    def get_theoretical_max(self) -> float: 
        """
        Based on arrival/departure rate, construct a bigraph with all possible edges and form maximal pairing.
        Return the maximal proportion of pairs that can be matched.
        """
        directed_graph = nx.DiGraph()
        for day in range(self.n_timesteps):
            arrivals_today = np.where(self.arrival_times == day)[0]

            for agent_idx in arrivals_today:
                directed_graph.add_node(agent_idx)

                for other_agent in directed_graph.nodes():
                    if other_agent != agent_idx:
                        if self.real_departure_times[other_agent] <= day: # is their departure time before my arrival time?
                            continue

                        if self.compatibility[agent_idx, other_agent] == 1:
                            directed_graph.add_edge(agent_idx, other_agent)
                        if self.compatibility[other_agent, agent_idx] == 1:
                            directed_graph.add_edge(other_agent, agent_idx)
        
        # Convert to undirected graph for matching (only including mutual compatibility)
        undirected_graph = nx.Graph()
        for u, v in directed_graph.edges():
            # Only add edge if there's mutual compatibility
            if directed_graph.has_edge(v, u):
                undirected_graph.add_edge(u, v)
        
        matching = nx.max_weight_matching(undirected_graph, maxcardinality=True)
        return (len(matching) * 2) / self.n_agents