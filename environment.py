import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time
import math
    
class PairedKidneyDonationEnv(gym.Env):
    def __init__(self, n_agents=1000, p=0.037, q=0.087, pct_hard=0.3, arrival_rate=1, criticality_rate=400, n_timesteps=700, use_cycles=False):
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

        self.action_space = MultiBinary(n_agents)

        self.compatibility = np.zeros((n_agents, n_agents))
        self.arrival_times = np.zeros(n_agents)
        self.real_departure_times = np.zeros(n_agents) # currently, we assume that each individual leaves the market after the same amount of time (w/ distribution)
        self.active_agents = np.zeros(n_agents) # 1 if the agent is in the market, 0 otherwise
        self.is_hard_to_match = np.zeros(n_agents) # 1 if the agent is hard to match, 0 otherwise
        self.matched_agents = np.zeros(n_agents) # 1 if the agent has been matched, 0 otherwise

        self.current_step = 0
        self.current_graph = nx.DiGraph()
        self.theoretical_max = 0
        self.use_cycles = use_cycles

        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        start_time = time.time()
        # Reset compatibility matrix
        self.compatibility = np.zeros((self.n_agents, self.n_agents))
        
        # Identify agents that are hard to match (e.g., highly sensitized patients)
        hard_indices = self.np_random.choice(self.n_agents, int(self.n_agents * self.pct_hard), replace=False)
        self.is_hard_to_match[hard_indices] = 1
        easy_indices = np.setdiff1d(np.arange(self.n_agents), hard_indices)

        random_values = self.np_random.random((self.n_agents, self.n_agents))

        # Fill compatibility matrix based on hard/easy matching criteria
        self.compatibility[np.ix_(hard_indices, easy_indices)] = random_values[np.ix_(hard_indices, easy_indices)] < self.p  # Hard-to-Easy
        self.compatibility[np.ix_(easy_indices, hard_indices)] = random_values[np.ix_(easy_indices, hard_indices)] < self.p  # Easy-to-Hard
        self.compatibility[np.ix_(easy_indices, easy_indices)] = random_values[np.ix_(easy_indices, easy_indices)] < self.q  # Easy-to-Easy
        self.compatibility[np.ix_(hard_indices, hard_indices)] = 0  # Hard-to-Hard
        self.compatibility[np.arange(self.n_agents), np.arange(self.n_agents)] = 0 # no self-compatibility

        # Generate arrivals using exponential distribution
        self.arrival_times = np.zeros(self.n_agents, dtype=int)
        inter_arrival_times = self.np_random.exponential(1.0/self.arrival_rate, self.n_agents)
        cumulative_times = np.cumsum(inter_arrival_times)
        scaled_times = (cumulative_times / np.max(cumulative_times) * (self.n_timesteps * 0.9)).astype(int)
        self.arrival_times = np.clip(scaled_times, 0, self.n_timesteps-1)
        self.arrival_times[0] = 0
        
        # Generate departure times based on arrival times plus criticality duration
        criticality_durations = self.np_random.exponential(self.criticality_rate, self.n_agents)
        self.real_departure_times = np.minimum((self.arrival_times + criticality_durations).astype(int), self.n_timesteps - 1)

        self.active_agents = np.zeros(self.n_agents)
        self.matched_agents = np.zeros(self.n_agents)
        end_time = time.time()

        self.current_step = 0
        self.current_graph = nx.DiGraph()
        for i in range(self.n_agents):
            self.current_graph.add_node(i)
        self.theoretical_max = self.get_theoretical_max()

        if options and "should_log" in options and options["should_log"]:
            print(f"Environment reset time: {end_time - start_time:.2f} seconds")

            print(f"Average arrival time: {np.mean(self.arrival_times):.2f}")
            print(f"Average criticality duration: {np.mean(criticality_durations):.2f}")

            print("Arrival times: ", self.arrival_times)
            print("Departure times: ", self.real_departure_times)

            print("Theoretical max: ", self.theoretical_max)



        return self.get_observation(), self.get_info()

    def start_over(self): # start over in a new environment with a similar setup
        self.current_step = 0
        self.current_graph = nx.DiGraph()
        for i in range(self.n_agents):
            self.current_graph.add_node(i)
        
        # Reset active agents and matched pairs
        self.active_agents = np.zeros(self.n_agents)
        for index, value in enumerate(self.arrival_times):
            if value == 0:
                self.active_agents[index] = 1
        self.matched_agents = np.zeros(self.n_agents)

        return self.get_observation(), self.get_info()

    def get_observation(self):
        return {
            "adjacency_matrix": nx.adjacency_matrix(self.current_graph).toarray() if self.current_graph else np.zeros((self.n_agents, self.n_agents)),
            "timestep": self.current_step,
            "arrivals": self.arrival_times,
            "departures": self.real_departure_times,
            "is_hard_to_match": self.is_hard_to_match,
            "active_agents": self.active_agents,
            "matched_agents": self.matched_agents,
            "total_timesteps": self.n_timesteps
        }
    def clear_node_edges(self, node):
        """Clear all edges connected to a specific node without removing the node."""
        if self.current_graph.has_node(node):
            # Get all edges connected to this node (both incoming and outgoing)
            edges_to_remove = list(self.current_graph.in_edges(node)) + list(self.current_graph.out_edges(node))
            # Remove all these edges at once
            self.current_graph.remove_edges_from(edges_to_remove)

    def step(self, action):
        previous_matched = np.sum(self.matched_agents)
        if action.sum() > 0:
            # check if the priority nodes should be matched
            priority_nodes = np.where(action == 1)[0]
            # Only proceed if there are selected priority nodes that are active
            if len(priority_nodes) > 0:
                # Create a subgraph with only the selected priority nodes and their neighbors
                selected_subgraph = nx.DiGraph()
                
                for node in priority_nodes:
                    if self.active_agents[node] == 1:  # Only consider active nodes
                        selected_subgraph.add_node(node)
                        for neighbor in self.current_graph.neighbors(node):
                            if self.active_agents[neighbor] == 1:  # Only consider active neighbors
                                selected_subgraph.add_edge(node, neighbor)
                                # Add the reverse edge if it exists
                                if self.current_graph.has_edge(neighbor, node):
                                    selected_subgraph.add_edge(neighbor, node)
            
                # Convert to undirected graph for matching
                undirected_subgraph = nx.Graph()
                for u, v in selected_subgraph.edges():
                    # Only add edge if there's mutual compatibility
                    if selected_subgraph.has_edge(v, u):
                        undirected_subgraph.add_edge(u, v)
            
                if self.use_cycles:
                    # Find the best matching in this graph
                    best_priority_matching = nx.max_weight_matching(undirected_subgraph, maxcardinality=True)
                
                    # Clear edges for matched nodes
                    for u, v in best_priority_matching:
                        self.clear_node_edges(u)
                        self.active_agents[u] = 0
                        self.matched_agents[u] = 1
                        self.clear_node_edges(v)
                        self.active_agents[v] = 0
                        self.matched_agents[v] = 1
                else:
                    # Find cycles in the selected subgraph
                    cycles = self.get_greedy_selected_cycles()
                    for cycle in cycles:
                        # Match all nodes in the cycle
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i + 1) % len(cycle)]
                            self.clear_node_edges(u)
                            self.active_agents[u] = 0
                            self.matched_agents[u] = 1
                            self.clear_node_edges(v)
                            self.active_agents[v] = 0
                            self.matched_agents[v] = 1

        # add the new arrivals to the graph 
        new_arrivals = np.where(self.arrival_times == self.current_step)[0]

        for agent_idx in new_arrivals:
            self.active_agents[agent_idx] = 1
            # Only connect to other ACTIVE agents
            for other_agent in self.current_graph.nodes():
                if other_agent != agent_idx and self.active_agents[other_agent] == 1:
                    if self.compatibility[agent_idx, other_agent] == 1:
                        self.current_graph.add_edge(agent_idx, other_agent)
                    if self.compatibility[other_agent, agent_idx] == 1:
                        self.current_graph.add_edge(other_agent, agent_idx)

        # Clear edges for departures
        departures = np.where(self.real_departure_times == self.current_step)[0]
        for agent_idx in departures:
            if self.active_agents[agent_idx] == 1:  # Only process active agents
                self.clear_node_edges(agent_idx)
                self.active_agents[agent_idx] = 0

        self.current_step += 1
        done = self.current_step == self.n_timesteps

        if self.theoretical_max > 0:
            reward = ((np.sum(self.matched_agents) - np.sum(previous_matched)) / self.n_agents) / self.theoretical_max
        else:
            print("Theoretical max is 0")
            reward = (np.sum(self.matched_agents) - np.sum(previous_matched)) / self.n_agents

        return self.get_observation(), reward, done, done, self.get_info()
    
    def get_info(self):
        num_hard_matched = sum([1 for i in range(self.n_agents) 
                            if self.is_hard_to_match[i] == 1 and self.matched_agents[i] == 1])
        num_hard_total = sum([1 for i in range(self.n_agents)
                        if self.is_hard_to_match[i] == 1 and self.arrival_times[i] <= self.current_step])
        
        num_regular_matched = sum([1 for i in range(self.n_agents)
                            if self.is_hard_to_match[i] == 0 and self.matched_agents[i] == 1])
        num_regular_total = sum([1 for i in range(self.n_agents)
                            if self.is_hard_to_match[i] == 0 and self.arrival_times[i] <= self.current_step])
        
        return {
            "hard_to_match_rate": num_hard_matched / max(1, num_hard_total),
            "regular_match_rate": num_regular_matched / max(1, num_regular_total),
            "active_agents": sum(self.active_agents),
            "total_matched": sum(self.matched_agents)
        }
        
    def render(self):
        plt.imshow(self.get_observation()["adjacency_matrix"])
        plt.show()

    def get_greedy_selected_cycles(self):
        cycles = []
        visited = set()
        for node in self.current_graph.nodes():
            if node not in visited:
                try:
                    # Find a cycle starting from the current node
                    cycle = nx.find_cycle(self.current_graph, source=node, orientation="original")
                    cycle_nodes = [edge[0] for edge in cycle] + [cycle[-1][1]]

                    # Check if the cycle is disjoint (no overlap with previously visited nodes)
                    if not any(n in visited for n in cycle_nodes):
                        cycles.append(cycle_nodes)
                        visited.update(cycle_nodes)
                except nx.NetworkXNoCycle:
                    # No cycle found starting from this node
                    continue
        return cycles

    def get_theoretical_max(self) -> float: 
        """
        Based on arrival/departure rate, construct a bigraph with all possible edges and form maximal pairing.
        Return the maximal proportion of pairs that can be matched.
        This is only for cycles of size 2
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