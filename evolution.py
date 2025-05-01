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
        total_score = reward

    return total_score / n_runs

# Wrapper for evaluating the main model
def evaluate_model_main(model: nn.Module, n_runs=8) -> float:
    model.eval()
    param_vector = nn.utils.parameters_to_vector(model.parameters()).detach()
    total_score = 0
    for _ in tqdm(range(n_runs)):
        total_score += evaluate_params(param_vector, n_runs=1)
    return total_score / n_runs

# Simplified initialization with high variance between models
def init_diverse_model(model_idx, hidden_dim):
    model = PKEModel(hidden_dim=hidden_dim)
    
    # Initialize with random parameters scaled by different factors
    # to ensure diversity among initial models
    scale = 5
    
    for name, param in model.named_parameters():
        # Use normal distribution with different variance per model
        std = 0.02 * (1 + model_idx % 3)
        nn.init.normal_(param, mean=0.0, std=std)
        
        # Apply scaling factor
        param.data *= scale
        
        # For some models, flip signs of parameters for even more diversity
        if model_idx % 2 == 1:
            param.data *= -1
            
    return model

# Add a new function to create a consensus model based on weighted averaging
def create_consensus_model(models: List[nn.Module], scores: List[float], hidden_dim: int) -> nn.Module:
    """
    Create a consensus model by averaging weights of models, weighted by their normalized scores.
    
    Args:
        models: List of models to combine
        scores: Performance scores for each model
        hidden_dim: Hidden dimension for the new model
        
    Returns:
        A new model with parameters set to weighted average of input models
    """
    # Create a new model to hold the consensus
    consensus_model = PKEModel(hidden_dim=hidden_dim)
    
    # Convert scores to numpy for easier manipulation
    scores_np = np.array(scores)
    
    # Normalize scores to create weights
    if np.std(scores_np) > 1e-6:
        # Convert to zero-mean and divide by standard deviation
        normalized_scores = (scores_np - np.mean(scores_np)) / np.std(scores_np)
        # Use softmax to get positive weights that sum to 1
        weights = np.exp(normalized_scores) / np.sum(np.exp(normalized_scores))
    else:
        # If all scores are identical, use equal weights
        weights = np.ones_like(scores_np) / len(scores_np)
    
    # Extract parameter vectors from all models
    param_vectors = []
    for model in models:
        param_vectors.append(nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy())
    
    # Create weighted average of parameters
    weighted_params = np.zeros_like(param_vectors[0])
    for i, param_vec in enumerate(param_vectors):
        weighted_params += weights[i] * param_vec
    
    # Convert back to tensor and set in the consensus model
    consensus_params = torch.tensor(weighted_params, dtype=torch.float32)
    nn.utils.vector_to_parameters(consensus_params, consensus_model.parameters())
    
    print(f"Created consensus model with weights: {weights}")
    return consensus_model

# Modify the step function to handle a batch of models at once
def step_population(models: List[nn.Module], n_perturb_per_model: int, perturb_size: float, 
                   n_runs: int, pool: mp.Pool, learning_rate: float) -> List[nn.Module]:
    """
    Perform evolutionary updates on an entire population of models simultaneously.
    
    Args:
        models: List of models in the population
        n_perturb_per_model: Number of perturbations to evaluate per model
        perturb_size: Size of perturbations
        n_runs: Number of evaluation runs per perturbation
        pool: Multiprocessing pool for parallel evaluation
        learning_rate: Step size for parameter updates
        
    Returns:
        Updated list of models
    """
    # Create all parameter vectors and perturbations in one batch
    all_param_vectors = []
    all_perturbations = []
    all_perturbation_indices = []  # Keep track of which model each perturbation belongs to
    
    for model_idx, model in enumerate(models):
        parameters = nn.utils.parameters_to_vector(model.parameters()).detach()
        num_params = parameters.shape[0]
        
        # Create perturbations for this model
        for i in range(n_perturb_per_model // 2):
            noise_vec = torch.randn(num_params) * perturb_size
            
            # Add both positive and negative perturbations
            all_param_vectors.append((parameters + noise_vec).detach())
            all_param_vectors.append((parameters - noise_vec).detach())
            
            all_perturbations.append(noise_vec)
            all_perturbations.append(-noise_vec)
            
            # Track which model these perturbations belong to
            all_perturbation_indices.extend([model_idx, model_idx])
    
    # Evaluate all perturbations in parallel
    print(f"Evaluating {len(all_param_vectors)} parameter sets across {pool._processes} workers...")
    eval_args = [(p_vec, n_runs) for p_vec in all_param_vectors]
    
    start_time = time.time()
    results = list(tqdm(pool.starmap(evaluate_params, eval_args), total=len(eval_args)))
    end_time = time.time()
    print(f"Batch evaluation took {end_time - start_time:.2f} seconds.")
    
    # Group results by model and apply updates
    updated_models = models.copy()
    for model_idx, model in enumerate(models):
        # Get this model's parameter vector
        parameters = nn.utils.parameters_to_vector(model.parameters()).detach()
        
        # Find perturbations and scores for this model
        model_perturbation_indices = [i for i, idx in enumerate(all_perturbation_indices) if idx == model_idx]
        model_perturbations = [all_perturbations[i] for i in model_perturbation_indices]
        model_scores = torch.tensor([results[i] for i in model_perturbation_indices])
        
        # Skip update if no evaluations for this model
        if len(model_scores) == 0:
            continue
            
        # Normalize scores for this model
        if model_scores.std() > 1e-6:
            normalized_scores = (model_scores - model_scores.mean()) / model_scores.std()
        else:
            normalized_scores = torch.zeros_like(model_scores)
            
        # Calculate update
        perturbation_tensor = torch.stack(model_perturbations)
        update = (normalized_scores.unsqueeze(1) * perturbation_tensor).sum(dim=0)
        parameters += learning_rate * update / (len(model_perturbations) * perturb_size)
        
        # Apply updated parameters
        nn.utils.vector_to_parameters(parameters, updated_models[model_idx].parameters())
    
    return updated_models

if __name__ == "__main__":
    # Initialize multiple diverse models
    num_models = 16
    models = []
    for i in range(num_models):
        model = init_diverse_model(i, HIDDEN_DIM)
        models.append(model)
        
    print(f"Initialized {num_models} different models with varied parameters")
    print(f"Total number of parameters per model: {sum(p.numel() for p in models[0].parameters())}")

    # Print initial parameter statistics
    for i, model in enumerate(models):
        parameters = nn.utils.parameters_to_vector(model.parameters())
        print(f"Model {i+1} initial parameter norm: {parameters.norm().item():.4f}")

    epochs = 50
    n_perturbations_total = mp.cpu_count() * 2 * num_models
    n_perturbations_per_model = max(2, (n_perturbations_total // num_models) // 2 * 2)  # Ensure it's even
    print(f"Total perturbations: {n_perturbations_total}, Per model: {n_perturbations_per_model}")
    
    n_eval_runs = 4
    n_final_eval_runs = 16
    initial_perturb_size = 0.4
    learning_rate = 0.01
    perturb_decay = 0.99
    
    # Configuration for consensus model creation
    consensus_interval = 5  # Create consensus model every N epochs
    consensus_top_k = 8     # Use top K models for consensus

    # Track performance of each model
    model_scores = [[] for _ in range(num_models)]
    
    # Verify initial diversity with a quick evaluation
    print("Evaluating initial model diversity...")
    initial_scores = []
    for i, model in enumerate(models):
        param_vector = nn.utils.parameters_to_vector(model.parameters()).detach()
        score = evaluate_params(param_vector, n_runs=2)
        initial_scores.append(score)
        print(f"Model {i+1} initial score: {score:.4f}")
    
    print(f"Initial score range: min={min(initial_scores):.4f}, max={max(initial_scores):.4f}")

    perturb_size = initial_perturb_size

    num_workers = mp.cpu_count()
    print(f"Using {num_workers} worker processes.")
    pool = mp.Pool(processes=num_workers)

    best_consensus_score = 0
    consensus_model = None

    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{epochs} - Training entire population as a batch")
            
            # Train all models in one batch
            models = step_population(
                models, 
                n_perturb_per_model=n_perturbations_per_model, 
                perturb_size=perturb_size, 
                n_runs=n_eval_runs, 
                pool=pool, 
                learning_rate=learning_rate
            )
            
            # Evaluate all models after training step
            current_epoch_scores = []
            for i, model in enumerate(models):
                param_vector = nn.utils.parameters_to_vector(model.parameters()).detach()
                epoch_score = evaluate_params(param_vector, n_runs=1)
                model_scores[i].append(epoch_score)
                current_epoch_scores.append(epoch_score)
            
            # Print current scores for all models
            print("\nModel scores after update:")
            for i, score in enumerate(current_epoch_scores):
                print(f"Model {i+1}: {score:.4f}")
            
            # Create consensus model at specified intervals
            if (epoch + 1) % consensus_interval == 0 or epoch == 0:
                print("\nCreating consensus model from top performing models...")
                
                # Get indices of top-k performing models
                top_k_indices = np.argsort(current_epoch_scores)[-consensus_top_k:]
                top_k_scores = [current_epoch_scores[i] for i in top_k_indices]
                top_k_models = [models[i] for i in top_k_indices]
                
                # Create consensus model
                new_consensus_model = create_consensus_model(top_k_models, top_k_scores, HIDDEN_DIM)
                
                # Evaluate the consensus model
                consensus_score = evaluate_model_main(new_consensus_model, n_runs=n_eval_runs)
                print(f"Consensus model score: {consensus_score:.4f}")
                
                # Replace worst performing model with the consensus model if it's good
                if consensus_score > best_consensus_score:
                    best_consensus_score = consensus_score
                    consensus_model = new_consensus_model
                    
                    # Find the worst performing model and replace it
                    worst_idx = np.argmin(current_epoch_scores)
                    print(f"Replacing Model {worst_idx+1} (score: {current_epoch_scores[worst_idx]:.4f}) with consensus model")
                    models[worst_idx] = consensus_model
                    model_scores[worst_idx].append(consensus_score)
                    current_epoch_scores[worst_idx] = consensus_score
                else:
                    print(f"Consensus model not better than previous best ({best_consensus_score:.4f})")
            
            perturb_size *= perturb_decay
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_end_time - epoch_start_time:.2f}s, New Perturbation size: {perturb_size:.4f}")
            
            # Find the best model so far
            best_model_idx = max(range(num_models), key=lambda i: current_epoch_scores[i])
            print(f"Best model this epoch: Model {best_model_idx+1} with score {current_epoch_scores[best_model_idx]:.4f}")
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                # Evaluate the best model more thoroughly
                best_model = models[best_model_idx]
                current_score = evaluate_model_main(best_model, n_runs=n_final_eval_runs // 2)
                print(f"Intermediate evaluation of best model: {current_score:.4f}")
                
                if consensus_model is not None:
                    consensus_score = evaluate_model_main(consensus_model, n_runs=n_final_eval_runs // 2)
                    print(f"Intermediate evaluation of consensus model: {consensus_score:.4f}")

    finally:
        pool.close()
        pool.join()

    print("Training finished.")
    
    # Find the best model overall by averaging the last 3 scores for stability
    final_avg_scores = []
    for scores in model_scores:
        if len(scores) >= 3:
            final_avg_scores.append(np.mean(scores[-3:]))
        else:
            final_avg_scores.append(np.mean(scores))
            
    best_model_idx = np.argmax(final_avg_scores)
    best_model = models[best_model_idx]
    
    print(f"Best model: Model {best_model_idx+1} with average recent score {final_avg_scores[best_model_idx]:.4f}")
    print("Score history:")
    for epoch, score in enumerate(model_scores[best_model_idx]):
        print(f"  Epoch {epoch+1}: {score:.4f}")
    
    # Final evaluation of both best individual model and consensus model
    final_best_score = evaluate_model_main(best_model, n_runs=n_final_eval_runs)
    print(f"Final Best Individual Model score after {epochs} epochs: {final_best_score:.4f}")
    
    if consensus_model is not None:
        final_consensus_score = evaluate_model_main(consensus_model, n_runs=n_final_eval_runs)
        print(f"Final Consensus Model score: {final_consensus_score:.4f}")
        
        # Save the overall best model
        if final_consensus_score > final_best_score:
            print("Consensus model is the overall best model")
            best_overall_model = consensus_model
            best_overall_score = final_consensus_score
        else:
            print("Individual model is the overall best model")
            best_overall_model = best_model
            best_overall_score = final_best_score
    else:
        best_overall_model = best_model
        best_overall_score = final_best_score
    
    print(f"Final Overall Best Model score: {best_overall_score:.4f}")