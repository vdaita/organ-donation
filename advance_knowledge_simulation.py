import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Optional
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from rich import print
import random
from slugify import slugify

seed = 24
random.seed(seed)
np.random.seed(seed)

num_envs = 8

n_agents = 100
n_timesteps = 128
death_time = 32
p = 0.037
q = 0.087
pct_hard = 0.6
warning_means = [1, 2, 3, 4]
# warning_means = [1]

title_name = f"n_agents={n_agents}, n_timesteps={n_timesteps}, death_time={death_time}, p={p}, q={q}, pct_hard={pct_hard}, n={num_envs}"
sluggified = slugify(title_name)

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
        warning_mean: int = 0, # when you know that someone is probably going to join the kidney exchange
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

        if seed and seed >= 0:
            self.reset(seed=seed)
        else:
            self.reset()
    
    def reset(self, seed: Optional[int] = None):
        if not seed:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed
        super().reset(seed=seed)

        # Steadily add agents so that all n_agents arrive evenly spaced over (n_timesteps - death_time)
        self.arrivals = np.linspace(0, self.n_timesteps - self.death_time, self.n_agents).astype(int)
        # print("Arrival times: ", self.arrivals)

        knowledge_time = self.np_random.exponential(1, size=self.n_agents) * self.warning_mean
        self.knowledge = np.clip(self.arrivals - knowledge_time, 0, self.n_timesteps - 1).astype(int)
        # print("Knowledge times: ", self.knowledge)

        time_in_exchange = self.np_random.exponential(self.death_time, size=self.n_agents)
        # print("Time in exchange: ", time_in_exchange)
        self.departures = np.clip(self.arrivals + time_in_exchange, 0, self.n_timesteps - 1).astype(int)
        # print("Departure times: ", self.departures)

        hard_indices = self.np_random.choice(self.n_agents, int(self.n_agents * self.pct_hard), replace=False)
        self.is_hard_to_match = np.zeros(self.n_agents)
        self.is_hard_to_match[hard_indices] = 1

        easy_indices = np.setdiff1d(np.arange(self.n_agents), hard_indices)

        self.compat = np.zeros((self.n_agents, self.n_agents), dtype=int)
        random_matrix = self.np_random.random((self.n_agents, self.n_agents))

        # print("Random matrix: ", random_matrix)

        self.compat[np.ix_(easy_indices, hard_indices)] = (random_matrix[np.ix_(easy_indices, hard_indices)] < self.p)
        self.compat[np.ix_(hard_indices, easy_indices)] = (random_matrix[np.ix_(hard_indices, easy_indices)] < self.p)
        self.compat[np.ix_(easy_indices, easy_indices)] = (random_matrix[np.ix_(easy_indices, easy_indices)] < self.q)
        self.compat[np.ix_(hard_indices, hard_indices)] = 0
        self.compat[np.diag_indices(self.n_agents)] = 0

        self.active_agents = np.zeros(self.n_agents)
        self.matched_agents = np.zeros(self.n_agents)
        self.time_matched = np.ones(self.n_agents) * -1

        self.current_step = 0

        self.current_graph = nx.Graph()
        self.node_indices = np.ones(self.n_agents, dtype=int) * -1 # why? i'd rather throw an error if the node is not in the graph rather than deal with unexpected behavior

        return self._get_observation(), {}

    def get_node_string(self, node):
        return f"node {node} (arrival: {self.arrivals[node]}, departure: {self.departures[node]}, knowledge: {self.knowledge[node]})"

    def step(self, action):
        # print("===========================")
        
        # matching
        matching_graph = self.current_graph.copy()
        knowledge_nodes = []
        if action == 1:
            known_inactive_nodes = np.where(
                (self.knowledge <= self.current_step) & 
                (self.arrivals > self.current_step) & 
                (self.departures > self.current_step)
            )[0]
            for node in known_inactive_nodes:
                for potential_match in range(self.n_agents):
                    if self.node_indices[potential_match] in self.current_graph.node_indices():
                        if self.compat[node, potential_match] == 1 and self.compat[potential_match, node] == 1:
                            node_range = (self.arrivals[node], self.departures[node])
                            potential_match_range = (self.arrivals[potential_match], self.departures[potential_match])
                            if do_ranges_overlap(node_range, potential_match_range):
                                self.node_indices[node] = matching_graph.add_node(node)
                                knowledge_nodes.append(node)
                                matching_graph.add_edge(self.node_indices[node], self.node_indices[potential_match], 1)
        
        matching = rx.max_weight_matching(matching_graph, max_cardinality=True)
        for a, b in matching:
            # a = np.where(self.node_indices == a_index)[0][0]
            # b = np.where(self.node_indices == b_index)[0][0]
            # print(f"Matching {self.get_node_string(a)} with {self.get_node_string(b)}")
            if self.active_agents[a] and self.active_agents[b]:
                nodes = [a, b]
                for node in nodes:
                    self.matched_agents[node] = 1
                    self.time_matched[node] = self.current_step
                    self.active_agents[node] = 0
                    self.current_graph.remove_node(self.node_indices[node])
                    self.node_indices[node] = -1
                print(f"Matched {self.get_node_string(a)} with {self.get_node_string(b)}")
            else:
                print(f"Cannot match {self.get_node_string(a)} with {self.get_node_string(b)} because one of them is inactive at {self.current_step}")
                ...
        
        # fix the node indices for the knowledge nodes
        self.node_indices[knowledge_nodes] = -1

        # arrivals
        arriving_nodes = np.where(self.arrivals == self.current_step)[0]
        for node in arriving_nodes:
            self.node_indices[node] = self.current_graph.add_node(node)
            self.active_agents[node] = 1
            for potential_match in range(self.n_agents):
                if self.node_indices[potential_match] in self.current_graph.node_indices():
                    if self.compat[node, potential_match] == 1 and self.compat[potential_match, node] == 1:
                        self.current_graph.add_edge(self.node_indices[node], self.node_indices[potential_match], 1)
                        print(f"Adding edge between {self.get_node_string(node)} and {self.get_node_string(potential_match)}")

        # departures
        departing_nodes = np.where(self.departures == self.current_step)[0]
        for node in departing_nodes:
            if not self.matched_agents[node]:
                self.active_agents[node] = 0
                self.matched_agents[node] = 0
                self.current_graph.remove_node(self.node_indices[node])
                self.node_indices[node] = -1
                print(f"Node {self.get_node_string(node)} departed at {self.current_step}")


        self.current_step += 1

        return self._get_observation(), self._get_reward(), self._is_done(), {}


    def _is_done(self):
        return self.current_step >= self.n_timesteps
    
    def _get_observation(self):
        return {
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
    
envs = [
    AdvanceKnowledgeSimulationEnvironment(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        death_time=death_time,
        p=p,
        q=q,
        pct_hard=pct_hard
    )
    for _ in range(num_envs)
]

rewards = {
    "regular": []
}

ratios = {
    "regular": []
}

for env in tqdm(envs):
    env.reset()

    regular_reward = env.compute_reward(use_waiting_graph=False)
    rewards["regular"].append(regular_reward)
    ratios["regular"].append(1)

    print(f"Regular reward: {regular_reward:.4f}")
    print("Matched values: ", np.where(env.matched_agents == 1)[0])

    for warning_mean in tqdm(warning_means, leave=False):
        env.warning_mean = warning_mean
        env.reset(seed=env.seed)

        if not f"warning_mean_{warning_mean}" in rewards:
            rewards[f"warning_mean_{warning_mean}"] = []
        if not f"warning_mean_{warning_mean}" in ratios:
            ratios[f"warning_mean_{warning_mean}"] = []

        waiting_reward = env.compute_reward(use_waiting_graph=True)
        print(f"Waiting reward for mean {warning_mean}: {waiting_reward:.4f}")
        rewards[f"warning_mean_{warning_mean}"].append(waiting_reward)
        ratios[f"warning_mean_{warning_mean}"].append(waiting_reward / regular_reward if regular_reward > 0 else 1)

        print("Matched values: ", np.where(env.matched_agents == 1)[0])

# Plotting the results
results_dir = "results/advance_knowledge"
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.boxplot(
    [rewards["regular"]] + [rewards[f"warning_mean_{wm}"] for wm in warning_means],
    labels=["regular"] + [f"warning={wm}" for wm in warning_means]
)
plt.ylabel("Reward")
plt.title("Reward Distribution for Different Warning Means")
plt.savefig(f"{results_dir}/reward_boxplot_{sluggified}.png")
plt.close()

# Plot the ratios
plt.figure(figsize=(10, 6))
plt.boxplot(
    [ratios["regular"]] + [ratios[f"warning_mean_{wm}"] for wm in warning_means],
    labels=["regular"] + [f"warning={wm}" for wm in warning_means]
)
plt.ylabel("Reward Ratio (waiting/regular)")
plt.title(f"Reward Ratio Distribution for Different Warning Means\n{title_name}")
plt.savefig(f"{results_dir}/ratio_boxplot_{sluggified}.png")
plt.show()
plt.close()