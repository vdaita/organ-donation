# Start off with a random decision-maker
import torch
from torch import nn
from environment import PairedKidneyDonationEnv
import numpy as np
from reinforce import PKEModel, convert_obs_to_tensors
from typing import List
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial
import time

# Define model architecture globally for worker processes
HIDDEN_DIM = 32
N_AGENTS = 100
N_TIMESTEPS = 32
DEATH_RANGE = [8, 12]

# Helper function for evaluation in worker processes
def evaluate_params(param_vector: torch.Tensor, n_runs: int) -> float:
    local_model = PKEModel(hidden_dim=HIDDEN_DIM)
    nn.utils.vector_to_parameters(param_vector, local_model.parameters())
    local_model.eval()

    env = PairedKidneyDonationEnv(
        n_agents=N_AGENTS,
        n_timesteps=N_TIMESTEPS,
        death_range=DEATH_RANGE
    )
    total_score = 0
    for _ in range(n_runs):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                agent_vecs, edge_index = convert_obs_to_tensors(obs, env.n_timesteps)
                action = local_model(agent_vecs, edge_index)
                action = action.detach().cpu().numpy()
            obs, reward, done, _, _ = env.step(action)
        greedy_score = env.get_greedy_percentage()
        if greedy_score > 0:
            total_score += (env.get_percentage()) / greedy_score

    return total_score / n_runs

# Wrapper for evaluating the main model
def evaluate_model_main(model: nn.Module, n_runs=8) -> float:
    model.eval()
    param_vector = nn.utils.parameters_to_vector(model.parameters()).detach()
    total_score = 0
    for _ in tqdm(range(n_runs)):
        total_score += evaluate_params(param_vector, n_runs=1)
    return total_score / n_runs

def step(model: nn.Module, n_perturb: int, perturb_size: float, n_runs: int, pool: mp.Pool, learning_rate: float) -> nn.Module:
    parameters = nn.utils.parameters_to_vector(model.parameters()).detach()
    num_params = parameters.shape[0]
    perturbations = torch.zeros((n_perturb + 1, num_params))
    scores = torch.zeros(n_perturb + 1)

    assert n_perturb % 2 == 0, "n_perturb must be even"

    for i in range(n_perturb // 2):
        noise_vec = torch.randn(num_params) * perturb_size
        perturbations[i * 2] = noise_vec
        perturbations[i * 2 + 1] = -noise_vec

    param_vectors_to_evaluate = [(parameters + noise_vec).detach() for noise_vec in perturbations]
    eval_args = [(p_vec, n_runs) for p_vec in param_vectors_to_evaluate]

    print(f"Evaluating {n_perturb + 1} parameter sets across {pool._processes} workers...")
    start_time = time.time()
    results = list(tqdm(pool.starmap(evaluate_params, eval_args), total=len(eval_args)))
    end_time = time.time()
    print(f"Evaluation took {end_time - start_time:.2f} seconds.")

    scores = torch.tensor(results)

    if scores.std() > 1e-6:
        normalized_scores = (scores - scores.mean()) / scores.std()
    else:
        normalized_scores = torch.zeros_like(scores)

    update = (normalized_scores.unsqueeze(1) * perturbations).sum(dim=0)
    parameters += learning_rate * update / (n_perturb * perturb_size)

    nn.utils.vector_to_parameters(parameters, model.parameters())
    return model

if __name__ == "__main__":
    model = PKEModel(hidden_dim=HIDDEN_DIM)
    model.reset_parameters()

    print(f"Total number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

    epochs = 50
    n_perturbations = mp.cpu_count() * 2
    if n_perturbations % 2 != 0: n_perturbations -= 1
    n_eval_runs = 2
    n_final_eval_runs = 16
    initial_perturb_size = 0.1
    learning_rate = 0.01
    perturb_decay = 0.99

    parameters = nn.utils.parameters_to_vector(model.parameters())
    print("Initial Parameter norm: ", parameters.norm().item())

    perturb_size = initial_perturb_size

    num_workers = mp.cpu_count()
    print(f"Using {num_workers} worker processes.")
    pool = mp.Pool(processes=num_workers)

    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model = step(model, n_perturb=n_perturbations, perturb_size=perturb_size, n_runs=n_eval_runs, pool=pool, learning_rate=learning_rate)
            perturb_size *= perturb_decay
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_end_time - epoch_start_time:.2f}s, New Perturbation size: {perturb_size:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                current_score = evaluate_model_main(model, n_runs=n_final_eval_runs // 2)
                print(f"Intermediate evaluation score: {current_score:.4f}")

    finally:
        pool.close()
        pool.join()

    print("Training finished.")
    final_score = evaluate_model_main(model, n_runs=n_final_eval_runs)
    print(f"Final Model score after {epochs} epochs: {final_score:.4f}")