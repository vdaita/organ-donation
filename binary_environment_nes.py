from binary_decision_environment import BinaryDecisionEnvironment
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

num_input_features = 9
num_envs = 8
lr = 0.1

binary_decision_model = nn.Sequential(
    nn.Linear(num_input_features, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

def make_env(rank):
    def _init():
        env = BinaryDecisionEnvironment(n_agents=250, n_timesteps=64)
        env.reset(seed=rank)
        return env
    return _init

def update_model(model, parameter_changes, rewards):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    parameters = nn.utils.parameters_to_vector(model.parameters())

    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    # print("New rewards: ", rewards)
    
    positive_indices = rewards > 0
    # print("Positive indices:", positive_indices)
    # print("Parameter changes shape:", parameter_changes.shape)
    
    parameter_changes = parameter_changes[positive_indices]
    rewards = rewards[positive_indices]
    weights = torch.softmax(rewards, dim=-1)
    weights = weights / weights.sum()
    weights = weights.unsqueeze(1)

    parameter_changes = parameter_changes * weights
    parameter_changes = parameter_changes.sum(dim=0)

    return parameters + parameter_changes * lr

def evaluate_model(model, seeds=[]):
    env = BinaryDecisionEnvironment(n_agents=250, n_timesteps=64)
    model_rewards = []

    for seed in tqdm(seeds, desc="Evaluating seeds", leave=False):
        env = BinaryDecisionEnvironment()
        obs, _ = env.reset(seed=seed)
        done = False

        while not done:
            obs = torch.tensor(obs, dtype=torch.float32)
            action = model(obs)
            obs, reward, done, _, _ = env.step(action > 0.5)

        model_rewards.append(np.sum(env.matched_agents) / env.n_agents)
    return np.array(model_rewards)

if __name__ == "__main__":
    parameters = nn.utils.parameters_to_vector(binary_decision_model.parameters())
    population_size = 32
    epochs = 16
    num_parameters = parameters.shape[0]

    parameters = torch.rand(num_parameters) * 2 - 1

    change_size = np.linspace(0.1, 0.5, population_size)
    change_size = torch.tensor(change_size, dtype=torch.float32)
    change_size = change_size.unsqueeze(1)
    
    for epoch in tqdm(range(epochs)):
        seeds = [int(s) for s in np.random.randint(0, 2 ** 32 - 1, size=8)]
        parameter_changes = torch.randn(population_size, num_parameters)
        parameter_changes = parameter_changes * change_size
        new_population = parameter_changes + parameters

        greedy_results = []
        patient_results = []
        for seed in tqdm(seeds):
            env = BinaryDecisionEnvironment()
            obs, _ = env.reset(seed=seed)
            greedy_results.append(env.get_greedy_result())
            patient_results.append(env.get_patient_result())

        greedy_results = torch.tensor(greedy_results)
        patient_results = torch.tensor(patient_results)

        # print("Greedy results:", greedy_results)
        # print("Patient results:", patient_results)

        results = []
        for i in tqdm(range(population_size), desc="Evaluating population", leave=False):
            new_parameters = new_population[i]
            torch.nn.utils.vector_to_parameters(new_parameters, binary_decision_model.parameters())
            result = evaluate_model(binary_decision_model, seeds=seeds)
            results.append(result)
        results = torch.tensor(results)

        greedy_ratio = results / greedy_results
        patient_ratio = results / patient_results
        mean_results = torch.mean(results, dim=-1)
        ratio = torch.mean(greedy_ratio, dim=-1)
        print("Greedy ratio: ", ratio)

        parameters = update_model(binary_decision_model, parameter_changes, mean_results)