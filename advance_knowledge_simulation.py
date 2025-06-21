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
from environment import PrioritySelectionPairedKidneyDonationEnv
from omegaconf import OmegaConf
from fire import Fire

seed = 24
random.seed(seed)
np.random.seed(seed)

def do_ranges_overlap(range1, range2):
    return not (range1[1] < range2[0] or range2[1] < range1[0])

class AdvanceKnowledgeSimulationEnvironment(PrioritySelectionPairedKidneyDonationEnv):
    def __init__(
        self,
        strategy: str,
        warning_mean: int = 0,
        **kwargs
    ):
        super(AdvanceKnowledgeSimulationEnvironment, self).__init__(**kwargs)
        self.strategy = strategy
        self.warning_mean = warning_mean

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        knowledge_time = self.np_random.exponential(1, size=self.n_agents) * self.warning_mean
        self.knowledge = np.clip(self.arrivals - knowledge_time, 0, self.n_timesteps - 1).astype(int)
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

                            if self.is_hard_to_match[node] == 1 or self.is_hard_to_match[other_node]:
                                weight += 1e-6

                            graph.add_edge(node, other_node, weight=weight)

        matching = nx.max_weight_matching(graph, maxcardinality=True)
        for node1, node2 in matching: # all of these nodes are unmatched, in the right timespace, and are a valid pair
            if self.strategy == "continuous":
                if self.arrivals[node1] <= self.current_step and self.arrivals[node2] <= self.current_step:
                    self.matched_agents[node1] = 1
                    self.matched_agents[node2] = 1
                    self.time_matched[node1] = self.current_step
                    self.time_matched[node2] = self.current_step
            elif self.strategy == "greedy":
                self.matched_agents[node1] = 1
                self.matched_agents[node2] = 1
                self.time_matched[node1] = np.max([self.current_step, self.arrivals[node1], self.arrivals[node2]])
                self.time_matched[node2] = np.max([self.current_step, self.arrivals[node1], self.arrivals[node2]])
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

def plot_stats(folder_name: str, filename: str, stat_name: str, data, warning_means):
    os.makedirs(folder_name, exist_ok=True)

    plt.figure(figsize=(18, 6))
    plt.boxplot(
        [data["regular"]] + [data[f"warning_mean_{wm}"] for wm in warning_means],
        labels=["regular"] + [f"warning={wm}" for wm in warning_means]
    )
    plt.ylabel(stat_name)
    plt.title(f"{stat_name} Distribution for Different Warning Means")
    plt.savefig(f"{folder_name}/{filename}.png")
    plt.close()

def main(config_file: str):
    config = OmegaConf.load(config_file)
    envs = [
        AdvanceKnowledgeSimulationEnvironment(
            n_agents=config.n_agents,
            n_timesteps=config.n_timesteps,
            death_time=config.death_time,
            p=config.p,
            q=config.q,
            pct_hard=config.pct_hard
        )
        for _ in range(config.num_envs)
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
        # print("Matched values: ", np.where(env.matched_agents == 1)[0])

        for warning_mean in tqdm(config.warning_means, leave=False):
            env.warning_mean = warning_mean
            env.reset(seed=env.seed)

            if not f"warning_mean_{warning_mean}" in rewards:
                rewards[f"warning_mean_{warning_mean}"] = []
                ratios[f"warning_mean_{warning_mean}"] = []
            waiting_reward = env.compute_reward(use_waiting_graph=True)
            print(f"Waiting reward for mean {warning_mean}: {waiting_reward:.4f}")
            rewards[f"warning_mean_{warning_mean}"].append(waiting_reward)
            ratios[f"warning_mean_{warning_mean}"].append(waiting_reward / regular_reward if regular_reward > 0 else 1)

    plot_stats(config.folder_name, "adv_knowledge_ratios.png", "Ratios", ratios, config.warning_means)
    plot_stats(config.folder_name, "adv_knowledge_rewards.png", "Rewards", rewards, config.warning_means)