import numpy as np
from binary_decision_environment import BinaryDecisionEnvironment
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

n_models = 1024
n_envs = 32

env = BinaryDecisionEnvironment(n_agents=100, n_timesteps=32, )
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

def evaluate_model(weight_seed, env_config, seeds, greedy_rewards):
    np.random.seed(weight_seed)
    # rather than randomly selecting the weight from a normal distribution, we want a more targeted approach
    # -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
    options = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    # we want to randomly select 9 weights from the options
    weight = np.random.choice(options, size=(9, 1))
    env = BinaryDecisionEnvironment(**env_config)
    model_rewards = [run_environment(env, weight, seed) for seed in seeds]
    ratios = np.array(model_rewards) / greedy_rewards
    ratio = np.mean(ratios)
    if ratio > 1:
        print("Ratio: ", ratio, "\nWeight: ", weight, "\nRatios: ", ratios)
    return ratio, weight

def run_environment(env, weight, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = np.dot(obs, weight) > 0.5
        obs, _, done, _, _ = env.step(action)
    return np.sum(env.matched_agents) / env.n_agents

if __name__ == '__main__':
    env_config = {'n_agents': 100, 'n_timesteps': 64}
    weight_seeds = range(n_models)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(partial(evaluate_model, env_config=env_config, 
                      seeds=seeds, greedy_rewards=greedy_rewards), weight_seeds), total=n_models))
    best_ratio, best_weight = max(results, key=lambda x: x[0])
    print(f"Best ratio: {best_ratio}, Best weight: {best_weight}")