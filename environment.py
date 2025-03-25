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
        
        self.n_days = self.n_days

        self.observation_space = Dict({
            "adjacency_matrix": MultiBinary([n_agents, n_agents]),
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
        self.compatibility = np.random.choice([0, 1], size=(self.n_agents, self.n_agents))

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

        return Dict(
            spaces={
                "adjacency_matrix": adj_matrix,
                "timestep": self.current_step
            }
        )
    
    def get_reward(self):
        return self.matched_pairs / self.n_agents

    def step(self, action):
        # check if the priority nodes should be matched
        if action["match_priority"] == 1:
            priority_nodes = np.where(action["selection"] == 1)[0]

            # create a helper graph where the only outgoing edges are from the selected nodes
            selected_direction_graph = nx.DiGraph()
            for node in priority_nodes:
                selected_direction_graph.add_node(node)
                for neighbor in self.current_graph.neighbors(node):
                    selected_direction_graph.add_edge(node, neighbor)
            # find the best matching in this graph
            best_priority_matching = nx.max_weight_matching(selected_direction_graph, maxcardinality=True)
            self.pairs += len(best_priority_matching)

            # remove the matched nodes from the graph
            for matched_pair in best_priority_matching:
                self.current_graph.remove_node(matched_pair[0])
                self.current_graph.remove_node(matched_pair[1])

        # check if the regular nodes should be matched
        if action["match_regular"] == 1:
            best_regular_matching = nx.max_weight_matching(self.current_graph, maxcardinality=True) # do the best at this timestep
            self.pairs += len(best_regular_matching)

            # remove the matched nodes from the graph
            for matched_node in best_regular_matching:
                self.current_graph.remove_node(matched_node[0])
                self.current_graph.remove_node(matched_node[1])

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

        return self.get_observation(), self.get_reward(), self.current_step == self.n_days, self.get_info()

    def get_theoretical_max(self) -> float: 
        """
        Based on arrival/departure rate, construct a bigraph with all possible edges and form maximal pairing.
        Return the maximal proportion of pairs that can be matched.
        """
        graph = nx.DiGraph()
        for day in range(self.n_days):
            arrivals_today = np.where(self.arrival_times == day)[0]

            for agent_idx in arrivals_today:
                graph.add_node(agent_idx)

                for other_agent in graph.nodes():
                    if other_agent != agent_idx:
                        
                        if self.real_departure_times[other_agent] <= day: # is their departure time before my arrival time?
                            continue
                                                
                        # if self.real_departure_times[agent_idx] <= self.arrival_times[other_agent]: # is my departure time before their arrival time - not important since it wouldn't have been added to graph yet
                        #    continue

                        if self.compatibility[agent_idx, other_agent] == 1:
                            graph.add_edge(agent_idx, other_agent)
                        if self.compatibility[other_agent, agent_idx] == 1:
                            graph.add_edge(other_agent, agent_idx)
        
        matching = nx.max_weight_matching(graph, maxcardinality=True)
        return (len(matching) * 2) / self.n_agents