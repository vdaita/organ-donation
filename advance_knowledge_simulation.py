import gymnasium as gym
import numpy as np
import rustworkx as rx
from typing import Optional
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

num_envs = 128
n_agents = 100
n_timesteps = 128
death_time = 64
p = 0.037
q = 0.087
pct_hard = 0.6
warning_means = [4, 8, 16, 32]

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

            nodes = [node1, node2]

            if self.current_step >= arrival_1 and self.current_step <= departure_1:
                if self.current_step >= arrival_2 and self.current_step <= departure_2:
                    for node in nodes:
                        self.matched_agents[node] = 1
                        self.active_agents[node] = 0
                        self.waiting_agents[node] = 0
                        self.time_matched[node] = self.current_step

                        self.waiting_graph.remove_node(node)
                        self.current_graph.remove_node(node)

        # check arrivals to the waiting graph
        waiting_arrivals = np.where(self.knowledge == self.current_step)[0]
        for waiting_arrival_node in waiting_arrivals:
            compatible_nodes = np.where(self.compat[waiting_arrival_node] == 1)[0]
            self.waiting_graph.add_node(waiting_arrival_node)
            for compat_node in compatible_nodes:
                if self.compat[compat_node, waiting_arrival_node] == 1:
                    waiting_node_range = [self.arrivals[waiting_arrival_node], self.departures[waiting_arrival_node]]
                    compat_node_range = [self.arrivals[compat_node], self.departures[compat_node]]
                    if self.waiting_agents[compat_node] == 1:
                        if do_ranges_overlap(waiting_node_range, compat_node_range):
                            self.waiting_graph.add_edge(waiting_arrival_node, compat_node)
            self.waiting_agents[waiting_arrival_node] = 1

        # check arrivals to the current graph
        arrivals = np.where(self.arrivals == self.current_step)[0] 
        for arrival_node in arrivals:
            compatible_nodes = np.where(self.compat[arrival_node] == 1)[0]
            self.current_graph.add_node(arrival_node)
            for compat_node in compatible_nodes:
                if self.compat[compat_node, arrival_node] == 1:
                    if self.active_agents[compat_node] == 1:
                        self.current_graph.add_edge(arrival_node, compat_node)
            self.active_agents[arrival_node] = 1

        # checking departures from the waiting and current graphs
        departures = np.where(self.departures == self.current_step)[0] 
        self.waiting_graph.remove_nodes_from([departure_node for departure_node in departures if self.waiting_agents[departure_node] == 1])
        self.current_graph.remove_nodes_from([departure_node for departure_node in departures if self.active_agents[departure_node] == 1])
        self.active_agents[departures] = 0
        self.waiting_agents[departures] = 0

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

    for warning_mean in tqdm(warning_means):
        env.warning_mean = warning_mean
        env.reset(seed=env.seed)

        if not f"warning_mean_{warning_mean}" in rewards:
            rewards[f"warning_mean_{warning_mean}"] = []
        if not f"warning_mean_{warning_mean}" in ratios:
            ratios[f"warning_mean_{warning_mean}"] = []

        waiting_reward = env.compute_reward(use_waiting_graph=True)
        rewards[f"warning_mean_{warning_mean}"].append(waiting_reward)
        ratios[f"warning_mean_{warning_mean}"].append(waiting_reward / regular_reward)

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
plt.savefig(f"{results_dir}/reward_boxplot.png")
plt.close()

# Plot the ratios
plt.figure(figsize=(10, 6))
plt.boxplot(
    [ratios["regular"]] + [ratios[f"warning_mean_{wm}"] for wm in warning_means],
    labels=["regular"] + [f"warning={wm}" for wm in warning_means]
)
plt.ylabel("Reward Ratio (waiting/regular)")
plt.title("Reward Ratio Distribution for Different Warning Means")
plt.savefig(f"{results_dir}/ratio_boxplot.png")
plt.close()