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

num_envs = 1

n_agents = 400
n_timesteps = 128
death_time = 48
p = 0.037
q = 0.087
pct_hard = 0.7
warning_means = [4, 8, 16, 24, 32, 48]

strategy = "continuous"
# could be: greedy
# could be: greedy with check

# warning_means = [1]

title_name = f"n_agents={n_agents}, n_timesteps={n_timesteps}, death_time={death_time}, p={p}, q={q}, pct_hard={pct_hard}, n={num_envs}, strategy={strategy}"
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

        self.matched_agents = np.zeros(self.n_agents)
        self.time_matched = np.ones(self.n_agents) * -1

        self.current_step = 0
        return self._get_observation(), {}

    def get_node_string(self, node):
        return f"node {node} (arrival: {self.arrivals[node]}, departure: {self.departures[node]}, knowledge: {self.knowledge[node]})"

    def step(self, action):
        if action == 0:
            relevant_nodes = np.where(
                (self.arrivals <= self.current_step) & 
                (self.departures >= self.current_step) & 
                (self.matched_agents == 0)
            )[0]
        elif action == 1:
            relevant_nodes = np.where(
                (self.knowledge <= self.current_step) &
                (self.departures >= self.current_step) & 
                (self.matched_agents == 0)
            )[0]

        graph = nx.Graph()
        for node in relevant_nodes:
            for other_node in relevant_nodes:
                if node == other_node:
                    continue
                other_range = (self.arrivals[other_node], self.departures[other_node])
                node_range = (self.arrivals[node], self.departures[node])
                if self.compat[node, other_node] == 1 and self.compat[other_node, node] == 1:
                        if do_ranges_overlap(node_range, other_range):
                            weight = 1
                            if other_range[0] < self.current_step and other_range[1] < self.current_step:
                                weight = 2
                            graph.add_edge(node, other_node, weight=weight)

        matching = nx.max_weight_matching(graph, maxcardinality=True)
        for node1, node2 in matching: # all of these nodes are unmatched, in the right timespace, and are a valid pair
            if strategy == "continuous":
                if self.arrivals[node1] <= self.current_step and self.arrivals[node2] <= self.current_step:
                    self.matched_agents[node1] = 1
                    self.matched_agents[node2] = 1
                    self.time_matched[node1] = self.current_step
                    self.time_matched[node2] = self.current_step
            
                # print(f"Matched: {self.get_node_string(node1)} and {self.get_node_string(node2)} in step: {self.current_step}")
            # else:
                # print(f"Waiting: {self.get_node_string(node1)} and {self.get_node_string(node2)} in step: {self.current_step}")

        self.current_step += 1
        return self._get_observation(), self._get_reward(), self._is_done(), {}


    def _is_done(self):
        return self.current_step >= self.n_timesteps
    
    def _get_observation(self):
        return {
            "matched_agents": self.matched_agents,
            "time_matched": self.time_matched,
            "current_step": self.current_step,
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
    
    def get_time_matched_hard(self):
        x = self.time_matched[self.is_hard_to_match == 1]
        return x[x > 0]

    def get_time_matched_easy(self):
        x = self.time_matched[self.is_hard_to_match == 0]
        return x[x > 0]
    
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

easy_matched_time = {
    "regular": []
}

hard_matched_time = {  
    "regular": []
}

for env in tqdm(envs):
    env.reset()

    regular_reward = env.compute_reward(use_waiting_graph=False)
    rewards["regular"].append(regular_reward)
    ratios["regular"].append(1)
    
    easy_matched_time["regular"].extend(env.get_time_matched_easy())
    hard_matched_time["regular"].extend(env.get_time_matched_hard())

    print(f"Regular reward: {regular_reward:.4f}")
    # print("Matched values: ", np.where(env.matched_agents == 1)[0])

    for warning_mean in tqdm(warning_means, leave=False):
        env.warning_mean = warning_mean
        env.reset(seed=env.seed)

        if not f"warning_mean_{warning_mean}" in rewards:
            rewards[f"warning_mean_{warning_mean}"] = []
            ratios[f"warning_mean_{warning_mean}"] = []
            easy_matched_time[f"warning_mean_{warning_mean}"] = []
            hard_matched_time[f"warning_mean_{warning_mean}"] = []

        waiting_reward = env.compute_reward(use_waiting_graph=True)
        print(f"Waiting reward for mean {warning_mean}: {waiting_reward:.4f}")
        rewards[f"warning_mean_{warning_mean}"].append(waiting_reward)
        ratios[f"warning_mean_{warning_mean}"].append(waiting_reward / regular_reward if regular_reward > 0 else 1)
        
        easy_matched_time[f"warning_mean_{warning_mean}"].extend(env.get_time_matched_easy())
        hard_matched_time[f"warning_mean_{warning_mean}"].extend(env.get_time_matched_hard())

        # print("Matched values: ", np.where(env.matched_agents == 1)[0])

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

plt.figure(figsize=(18, 8))
easy_data = []
hard_data = []
easy_labels = []
hard_labels = []

for key in ["regular"] + [f"warning_mean_{wm}" for wm in warning_means]:
    if key == "regular":
        easy_label = "Easy (No Warning)"
        hard_label = "Hard (No Warning)"
    else:
        wm = key.replace("warning_mean_", "")
        easy_label = f"Easy (Warning={wm})"
        hard_label = f"Hard (Warning={wm})"
    easy_data.append(easy_matched_time[key])
    easy_labels.append(easy_label)
    hard_data.append(hard_matched_time[key])
    hard_labels.append(hard_label)

# Combine easy and hard data for side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

axes[0].boxplot(easy_data, labels=easy_labels)
axes[0].set_title("Time to Match Distribution (Easy Cases)")
axes[0].set_ylabel("Time to Match")
axes[0].tick_params(axis='x', rotation=45)

axes[1].boxplot(hard_data, labels=hard_labels)
axes[1].set_title("Time to Match Distribution (Hard Cases)")
axes[1].tick_params(axis='x', rotation=45)

fig.suptitle("Time to Match Distribution for Easy and Hard Cases")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{results_dir}/time_to_match_boxplot_{sluggified}.png")
plt.show()