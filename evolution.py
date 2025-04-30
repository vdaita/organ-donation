# Start off with a random decision-maker
import torch
from torch import nn
from environment import PairedKidneyDonationEnv
import numpy as np
from reinforce import PKEModel, convert_obs_to_tensors
from typing import List
from tqdm import tqdm
import torch.multiprocessing as mp

model = PKEModel(hidden_dim=32)
print(f"Total number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

def evaluate_model(model: nn.Module, n_runs=8) -> float:
    env = PairedKidneyDonationEnv(
        n_agents=100,
        n_timesteps=32,
        death_range=[8, 12]
    )
    total_reward = 0
    for _ in tqdm(range(n_runs)):
        obs, _ = env.reset()
        done = False
        while not done:
            agent_vecs, edge_index = convert_obs_to_tensors(obs, env.n_timesteps)
            action = model(agent_vecs, edge_index)
            action = action.detach().cpu().numpy()
            obs, reward, done, _, _ = env.step(action)
        total_reward += (env.get_percentage()) / (env.get_greedy_percentage())
    return total_reward / n_runs

def step(model: nn.Module, n_perturb: int, perturb_size: float, n_runs: int) -> nn.Module:
    parameters = nn.utils.parameters_to_vector(model.parameters())
    perturbations = torch.zeros((n_perturb + 1, parameters.shape[0]))
    scores = torch.zeros(n_perturb + 1)

    assert n_perturb % 2 == 0, "n_perturb must be even"

    for i in range(n_perturb // 2):
        noise_vec = torch.zeros_like(parameters)
        noise_vec.normal_(mean=0, std=perturb_size)
        perturbations[i * 2] = noise_vec
        
        noise_vec = noise_vec.clone()
        noise_vec *= -1
        perturbations[i * 2 + 1] = noise_vec

    for i, noise_vec in enumerate(perturbations): # include the 0 vector as a perturbation
        nn.utils.vector_to_parameters(parameters + noise_vec, model.parameters())
        scores[i] = evaluate_model(model, n_runs=n_runs)
        perturbations[i] = noise_vec

    normalized_scores = (scores - scores.mean()) / scores.std()
    # print("Normalized score values: ", normalized_scores)
    normalized_scores = normalized_scores.unsqueeze(1)
    # print("Normalized scores: ", normalized_scores.shape)
    # print("Perturbations shape: ", perturbations.shape)
    update = normalized_scores * perturbations
    # print("Update shape: ", update.shape)
    parameters += update.sum(dim=0)
    nn.utils.vector_to_parameters(parameters, model.parameters())
    return model

model.reset_parameters()
epochs = 100
parameters = nn.utils.parameters_to_vector(model.parameters()) 
print("Parameters: ", parameters)
parameter_norm = (parameters ** 2).sum().sqrt()
print("Parameter norm: ", parameter_norm.item())
perturb_size = 0.5 * parameter_norm.item()
decrease_factor = 0.98

for epoch in range(epochs):
    model = step(model, n_perturb=8, perturb_size=perturb_size, n_runs=4)
    perturb_size *= decrease_factor
    print(f"Epoch {epoch + 1}/{epochs}, Perturbation size: {perturb_size}")
    print(f"Model score: {evaluate_model(model)}")