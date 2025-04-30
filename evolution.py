# Start off with a random decision-maker
import torch
from torch import nn
from environment import PairedKidneyDonationEnv
import numpy as np
from reinforce import PKEModel
from typing import List
from tqdm import tqdm

model = PKEModel()
print(f"Total number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

def evaluate_model(model: nn.Module, n_runs=32) -> float:
    env = PairedKidneyDonationEnv()
    total_reward = 0
    for _ in range(n_runs):
        obs = env.reset()
        done = False
        while not done:
            action = model(obs)
            obs, reward, done, _ = env.step(action)
        total_reward += (env.get_percentage()) / (env.get_greedy_percentage())
    return total_reward / n_runs

def step(model: nn.Module, n_perturb: int, perturb_size: float) -> nn.Module:
    parameters = nn.utils.parameters_to_vector(model.parameters())
    perturbations = torch.zeros((n_perturb + 1, parameters.shape[0]))
    scores = torch.zeros(n_perturb + 1)

    for i in range(n_perturb):
        noise_vec = torch.zeros_like(parameters)
        noise_vec.normal_(mean=0, std=perturb_size)
        nn.utils.vector_to_parameters(parameters + noise_vec, model.parameters())
        scores[i] = evaluate_model(model)
        perturbations[i] = noise_vec

    nn.utils.vector_to_parameters(parameters, model.parameters())
    scores[n_perturb] = evaluate_model(model)

    normalized_scores = (scores - scores.mean()) / scores.std()
    update = normalized_scores * perturbations
    parameters += update
    nn.utils.vector_to_parameters(parameters, model.parameters())
    return model

model.reset_parameters()
epochs = 100
parameter_norm = (nn.utils.parameters_to_vector(model.parameters()) ** 2).sum().sqrt()
perturb_size = 0.5 * parameter_norm.item()
decrease_factor = 0.98

for epoch in range(epochs):
    model = step(model, n_perturb=32, perturb_size=perturb_size)
    perturb_size *= decrease_factor
    print(f"Epoch {epoch + 1}/{epochs}, Perturbation size: {perturb_size.item()}")
    print(f"Model score: {evaluate_model(model)}")