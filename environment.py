import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
import matplotlib.pyplot as plt

class PairedKidneyDonationEnv(gym.Env):
    def __init__(self, n_agents=1000, hard_match_pct=0.2, arrival_rate=1, departure_rate=1, criticality_rate=400, n_timesteps=700):
        self.n_agents = n_agents

        self.hard_match_pct = hard_match_pct
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
        self.compatibility = np.random.choice([0, 1], size=(self.n_agents, self.n_agents)) # make some agents hard to match as well

        self.arrival_times = np.random.poisson(self.arrival_rate, size=self.n_agents)
        self.real_departure_times = np.random.poisson(self.criticality_rate, size=self.n_agents)

        self.active_agents = np.ones(self.n_agents)

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
                    if self.current_graph.has_node(v):  # Check before removing
                        self.current_graph.remove_node(v)

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
                if self.current_graph.has_node(v):  # Check before removing
                    self.current_graph.remove_node(v)

        # add the new arrivals to the graph 
        new_arrivals = np.where(self.arrival_times == self.current_step)[0]

        for agent_idx in new_arrivals:
            self.current_graph.add_node(agent_idx)
            for other_agent in self.current_graph.nodes():
                if other_agent != agent_idx:
                    if self.compatibility[agent_idx, other_agent] == 1:
                        self.current_graph.add_edge(agent_idx, other_agent)
                    if self.compatibility[other_agent, agent_idx] == 1:
                        self.current_graph.add_edge(other_agent, agent_idx)

        # remove the old departures
        departures = np.where(self.real_departure_times == self.current_step)[0]
        for agent_idx in departures:
            self.current_graph.remove_node(agent_idx)
            self.active_agents[agent_idx]

        self.current_step += 1

        return self.get_observation(), self.get_reward(), self.current_step == self.n_timesteps, self.get_info()

    def get_info(self):
        return {}
    
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