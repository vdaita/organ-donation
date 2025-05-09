import gymnasium as gym
import numpy as np
import rustworkx as rx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time
import copy
import numba as nb

class PairedKidneyDonationEnv(gym.Env):
    def __init__(self, n_agents=1000, p=0.037, q=0.087, pct_hard=0.6, arrival_rate=1, death_time=350, n_timesteps=700, use_cycles=False, seed=-1, greedy_comp_mode=True):
        self.n_agents = n_agents

        self.p = p
        self.q = q
        self.pct_hard = pct_hard
        self.arrival_rate = arrival_rate
        self.death_time = death_time # going back to the exponential distribution for departure
        self.use_cycles = use_cycles
        self.n_timesteps = n_timesteps
        self.observation_space = Dict({
            "adjacency": MultiBinary((n_agents, n_agents)),
            "timestep": Discrete(n_timesteps),
            "arrivals": Box(low=0, high=n_timesteps, shape=(n_agents,), dtype=np.int32),
            "departures": Box(low=0, high=float('inf'), shape=(n_agents,), dtype=np.float32),
            "is_hard": MultiBinary(n_agents),
            "active": MultiBinary(n_agents),
            "matched": MultiBinary(n_agents),
            "total_timesteps": Discrete(n_timesteps + 1)  # +1 because it includes n_timesteps
        })
        # Change action space from nodes to edges (adjacency matrix)
        self.action_space = Box(low=-1, high=1, shape=(n_agents, n_agents))
        self.seed = seed
        self.greedy_comp_mode = greedy_comp_mode 
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
        exponential_distributions = self.np_random.exponential(1, (self.n_agents, )) * self.death_time
        exponential_distributions = exponential_distributions.astype(int)
        self.real_departure_times = np.minimum(self.arrival_times + exponential_distributions, np.ones(self.n_agents) * self.n_timesteps)
        # print("Exponential distributions: ", exponential_distributions)
        # print("Departure times: ", self.real_departure_times)

        self.active_agents = np.zeros(self.n_agents)
        self.matched_agents = np.zeros(self.n_agents)
        self.time_matched = np.ones(self.n_agents) * -1
        end_time = time.time()

        if options and "should_log" in options and options["should_log"]:
            print(f"Environment reset time: {end_time - start_time:.2f} seconds")

            print(f"Average arrival time: {np.mean(self.arrival_times):.2f}")
            print("Arrival times: ", self.arrival_times)
            print("Departure times: ", self.real_departure_times)
            print("Exponential distributions: ", exponential_distributions)

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

    def step(self, action, is_greedy=False):
        previous_matched = np.copy(self.matched_agents)
        adj_matrix = rx.adjacency_matrix(self.current_graph)
        # make sure that each value is valid, only
        valid_edges = adj_matrix * np.outer(self.active_agents, self.active_agents)
        selected_edges = valid_edges * action
        
        if np.sum(selected_edges) > 0:
            if is_greedy:
                if self.use_cycles:
                    raise NotImplementedError("Greedy cycles not implemented anymore")
                reg_graph = rx.PyGraph()
                for u in range(self.n_agents):
                    for v in range(self.n_agents):
                        if selected_edges[u, v] > 0 and self.active_agents[u] == 1 and self.active_agents[v] == 1:
                            reg_graph.add_edge(u, v)
                best_matching = rx.max_weight_matching(reg_graph, maxcardinality=True)
                for u, v in best_matching:
                    self.node_matched(u)
                    self.node_matched(v)
            else:
                # currently, only consider one edge at a time, in order
                # interesting locations
                rows, cols = np.where(selected_edges > 0)
                if len(rows) > 0:
                    edge_weights = selected_edges[rows, cols]
                    edges = np.column_stack((edge_weights, rows, cols))
                    edges = edges[edges[:, 0].argsort()[::-1]]
                    for _, u, v in edges:
                        u, v = int(u), int(v)
                        if self.active_agents[u] == 1 and self.active_agents[v] == 1 and self.current_graph.has_edge(u, v):
                            self.node_matched(u)
                            self.node_matched(v)

        self.manage_arrivals_departures()

        self.current_step += 1
        done = self.current_step == self.n_timesteps

        unmatched_departures = np.sum((self.real_departure_times == self.current_step) * (1 - self.matched_agents)) / self.n_agents
        reward = (-unmatched_departures) * (0.5)
        matched_now = np.sum(self.matched_agents - previous_matched) / self.n_agents

        hard_matched_now = np.sum((self.matched_agents - previous_matched) * self.is_hard_to_match) / max(1, np.sum(self.is_hard_to_match))
        regular_matched_now = np.sum((self.matched_agents - previous_matched) * (1 - self.is_hard_to_match)) / max(1, np.sum(1 - self.is_hard_to_match))
        reward += (hard_matched_now * 2) + (regular_matched_now * 0.5)
        # reward = matched_now
        # if done:
        #     if not is_greedy:
        #         _, greedy_pct = self.get_greedy_percentage()
        #         model_pct = np.sum(self.matched_agents) / self.n_agents
        #         improvement = model_pct - greedy_pct
        #         if improvement > 0:
        #             reward += improvement * 3

        # if not is_greedy:
        #     reward += self.greedy_future(action) * 0.5 # maximize the future reward

        # reward = 0
        # if done:
        #     if not is_greedy:
        #         my_reward = np.sum(self.matched_agents) / self.n_agents
        #         greedy_reward = self.get_greedy_percentage()
        #         reward = my_reward / greedy_reward

        return self.get_observation(), reward, done, done, self.get_info()
    
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

    def greedy_future(self, action):
        # each action needs to optimize the value that happens in the future
        copied_env = copy.deepcopy(self)
        obs, info = copied_env.start_over()
        done = copied_env.current_step >= self.n_timesteps
        future_rewards = 0
        while not done:
            action = np.ones((self.n_agents, self.n_agents))
            obs, reward, done, _, info = copied_env.step(action, is_greedy=True)
            future_rewards += reward
        return future_rewards

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
            "hard_percentage": sum(self.is_hard_to_match) / self.n_agents,
            "hard_to_match_rate": num_hard_matched / max(1, num_hard_total),
            "regular_match_rate": num_regular_matched / max(1, num_regular_total),
            "active_agents": sum(self.active_agents),
            "total_matched": sum(self.matched_agents)
        }

    def print_info(self, info):
        print(f"Active agents: {info['active_agents']}")
        print(f"Hard to match percentage: {info['hard_percentage']:.2f}")
        print(f"Total matched: {info['total_matched']}")
        print(f"Hard to match rate: {info['hard_to_match_rate']:.2f}")
        print(f"Regular match rate: {info['regular_match_rate']:.2f}")
        print(f"Current step: {self.current_step}/{self.n_timesteps}")
        
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
    
    def get_waiting_time_stats(self):
        # return a dictionary describing the waiting time of the elements
        result = {
            "by_difficulty": []
        }
        for difficulty in [0, 1]:
            relevant_indices = np.where(self.is_hard_to_match == difficulty)[0]
            relevant_arrivals = self.arrival_times[relevant_indices]
            relevant_time_matched = self.time_matched[relevant_indices]
            if len(relevant_indices) > 0:
                waiting_times = relevant_time_matched - relevant_arrivals
                waiting_times[relevant_time_matched < 0] = 0
                avg_waiting_time = np.mean(waiting_times)
                std_waiting_time = np.std(waiting_times)
                pct_matched = sum(self.matched_agents[relevant_indices]) / len(relevant_indices)
            else:
                pct_matched = 0
                avg_waiting_time = -1.0
                std_waiting_time = -1.0

            result["by_difficulty"].append({
                "difficulty": difficulty,
                "avg_waiting_time": avg_waiting_time,
                "std_waiting_time": std_waiting_time,
                "pct_matched": pct_matched
            })
        return result
    
    def print_waiting_time_stats(self, waiting_time_stats):
        for stat in waiting_time_stats["by_difficulty"]:
            difficulty = "Hard to Match" if stat["difficulty"] == 1 else "Easy to Match"
            print(f"{difficulty}:")
            print(f"  Average Waiting Time: {stat['avg_waiting_time']:.2f}")
            print(f"  Standard Deviation of Waiting Time: {stat['std_waiting_time']:.2f}")
            print(f"  Percentage Matched: {stat['pct_matched'] * 100:.2f}%")

    def get_greedy_percentage(self):
        obs, info = self.start_over()
        reward, done = 0, False
        # Add counters
        total_departures = 0
        capped_departures = 0
        
        # Calculate original departure distribution
        uncapped_departures = self.arrival_times + (self.np_random.exponential(1, (self.n_agents, )) * self.death_time)
        capped_departures_count = np.sum(uncapped_departures >= self.n_timesteps)
        # print(f"Agents with departures capped by simulation end: {capped_departures_count}/{self.n_agents} ({capped_departures_count/self.n_agents*100:.1f}%)")
        
        while not done:
            action = np.outer(self.active_agents, self.active_agents)
            obs, new_reward, done, _, info = self.step(action, is_greedy=False)
            reward += new_reward
        return (np.sum(self.matched_agents) / self.n_agents)
    

    def get_percentage(self):
        return sum(self.matched_agents) / self.n_agents
    
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
def get_edges(selected_nodes, matchable_nodes, adj_matrix):
    adj_bidirectional = np.logical_and(adj_matrix, adj_matrix.T)
    adj_view = adj_bidirectional[selected_nodes][:, matchable_nodes]
    edges = np.argwhere(adj_view == 1)
    edges = [(selected_nodes[edge[0]], matchable_nodes[edge[1]], 1) for edge in edges]
    return edges

class PrioritySelectionPairedKidneyDonationEnv(PairedKidneyDonationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.MultiBinary(self.n_agents)

    def match_subgraph(self, selected_nodes, matchable_nodes, adj_matrix):
        start_time = time.perf_counter()
        
        if self.use_cycles:
            graph = rx.PyDiGraph()
            graph.add_nodes_from(range(self.n_agents))
            rx_edges = get_edges_cycles(selected_nodes, matchable_nodes, adj_matrix)
            graph.add_edges_from(rx_edges)

            edge_add_time = time.perf_counter() - start_time

            while True:
                current_cycle = rx.digraph_find_cycle(graph)
                if current_cycle is None or len(current_cycle) == 0:
                    break
                for u, v in current_cycle:
                    self.node_matched(u)
                    self.node_matched(v)
                    graph.remove_node(u)
                    graph.remove_node(v)
        else:
            graph = rx.PyGraph()
            graph.add_nodes_from(range(self.n_agents))
            rx_edges = get_edges(selected_nodes, matchable_nodes, adj_matrix)
            graph.add_edges_from(rx_edges)

            edge_add_time = time.perf_counter() - start_time

            # Find the maximum matching
            best_matching = rx.max_weight_matching(graph, max_cardinality=True)
            for u, v in best_matching: # we already know they are getting matched
                self.node_matched(u)
                self.node_matched(v)

        matching_time = time.perf_counter() - start_time - edge_add_time
        end_time = time.perf_counter() - start_time - edge_add_time - matching_time

        # print(f"Adjacency matrix time: {adj_matrix_time:.6f} seconds")
        # print(f"Edge addition time: {edge_add_time:.6f} seconds")
        # print(f"Matching time: {matching_time:.6f} seconds")
        # print(f"Marking time: {end_time:.6f} seconds")
       

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

    def get_greedy_percentage(self):
        obs, _ = self.start_over()
        done = False
        while not done:
            action = np.ones(self.n_agents)
            obs, reward, done, _, _ = self.step(action)
        total_reward = np.sum(self.matched_agents) / self.n_agents
        return total_reward

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
        return total_reward
    
    def get_hard_waiting_time(self):
        hard_waiting_times = self.time_matched[self.is_hard_to_match == 1] - self.arrival_times[self.is_hard_to_match == 1]
        return hard_waiting_times[hard_waiting_times > 0]
    
    def get_easy_waiting_time(self):
        easy_waiting_times = self.time_matched[self.is_hard_to_match == 0] - self.arrival_times[self.is_hard_to_match == 0]
        return easy_waiting_times[easy_waiting_times > 0]