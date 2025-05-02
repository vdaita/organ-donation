import numpy as np
from binary_decision_environment import BinaryDecisionEnvironment
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

n_models = 4000
n_envs = 16

env = BinaryDecisionEnvironment(n_agents=100, n_timesteps=64)
seeds = np.random.randint(0, 2 ** 32 - 1, size=(n_envs, 1))
seeds = [int(seed) for seed in seeds.flatten()]

greedy_rewards = []
patient_rewards = []
for j in tqdm(seeds, desc="Environments"):
    obs, _ = env.reset(seed=j)
    greedy_rewards.append(env.get_greedy_result())
    patient_rewards.append(env.get_patient_result())
greedy_rewards = np.array(greedy_rewards)
patient_rewards = np.array(patient_rewards)

best_greedy_ratio = 0
best_weight = None
best_bias = None

tested_weights = set()

for i in tqdm(range(n_models), desc="Models", leave=False):
    options = [0, 0.5, 1]
    while True:
        weight = np.zeros((9, 1))
        non_zero_indices = np.random.choice(9, size=5, replace=False)
        weight[non_zero_indices] = np.random.choice([0.5, 1], size=5, replace=True).reshape(-1, 1)
        weight_tuple = tuple(weight.flatten())
        if weight_tuple not in tested_weights:
            tested_weights.add(weight_tuple)
            break

    model_rewards = []
    for j in tqdm(seeds, desc="Environments", leave=False):
        obs, _ = env.reset(seed=j)
        done = False
        while not done:
            action = np.dot(obs, weight)
            obs, reward, done, _, _ = env.step(action >= 1)
        model_rewards.append(np.sum(env.matched_agents) / env.n_agents)
    model_rewards = np.array(model_rewards)

    ratios = model_rewards / greedy_rewards
    ratio = np.mean(ratios)

    if ratio > best_greedy_ratio:
        best_greedy_ratio = ratio
        print(f"New best ratio: {best_greedy_ratio}")
        print(f"Ratios: {ratios}")
        print(f"Weight: {weight}")
        best_weight = weight