import torch
from torch import nn
from environment import PairedKidneyDonationEnv
import numpy as np
from reinforce import PKEModel, convert_obs_to_tensors
from tqdm import tqdm
import torch.multiprocessing as mp
import time
import os
from datetime import datetime

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

def step(model: nn.Module, n_perturb: int, perturb_size: float, n_runs: int, pool: mp.Pool, learning_rate: float) -> nn.Module:
    parameters = nn.utils.parameters_to_vector(model.parameters()).detach()
    num_params = parameters.shape[0]
    perturbations = torch.zeros((n_perturb + 1, num_params))
    
    # Make sure n_perturb is even for antithetic sampling
    assert n_perturb % 2 == 0, "n_perturb must be even"

    # Generate random perturbations
    # First perturbation is zero for evaluating the unperturbed model
    for i in range(1, n_perturb + 1, 2):
        # Generate random noise
        noise_vec = torch.randn(num_params)
        # Normalize the noise vector to maintain unit variance
        noise_vec = noise_vec / noise_vec.norm() * perturb_size
        
        # Antithetic sampling (mirrored perturbations)
        perturbations[i] = noise_vec
        perturbations[i+1] = -noise_vec
    
    # Create parameter vectors for each perturbation
    param_vectors_to_evaluate = [(parameters + noise_vec).detach() for noise_vec in perturbations]
    eval_args = [(p_vec, n_runs) for p_vec in param_vectors_to_evaluate]

    print(f"Evaluating {n_perturb + 1} parameter sets across {pool._processes} workers...")
    start_time = time.time()
    results = list(tqdm(pool.starmap(evaluate_params, eval_args), total=len(eval_args)))
    end_time = time.time()
    print(f"Evaluation took {end_time - start_time:.2f} seconds.")

    # Convert results to tensor
    fitness_scores = torch.tensor(results)
    
    # Print raw scores for debugging
    print(f"Raw fitness scores: {fitness_scores}")
    
    # Proper fitness shaping for NES - using rank-based normalization
    # NES typically uses fitness ranking rather than raw score normalization
    sorted_indices = torch.argsort(fitness_scores)
    ranked_weights = torch.zeros_like(fitness_scores)
    
    # Apply rank-based weights - linear weighting
    for i, idx in enumerate(sorted_indices):
        ranked_weights[idx] = i / (n_perturb)
    
    # Center the weights
    ranked_weights = ranked_weights - torch.mean(ranked_weights)
    
    # Scale weights for numerical stability
    if torch.std(ranked_weights) > 1e-6:
        ranked_weights = ranked_weights / torch.std(ranked_weights)
    
    print(f"Ranked weights: {ranked_weights}")
    
    # Compute the gradient approximation using NES formula
    # Skip the first perturbation (unperturbed model) in the gradient estimation
    gradient = torch.zeros_like(parameters)
    for i in range(1, n_perturb + 1):
        gradient += ranked_weights[i] * perturbations[i]
    
    # Scale gradient by number of perturbations and perturbation size
    gradient = gradient / (n_perturb * perturb_size)
    
    # Update parameters using the estimated gradient
    parameters += learning_rate * gradient
    
    # Apply the updated parameters to the model
    nn.utils.vector_to_parameters(parameters, model.parameters())
    
    # Return the updated model
    return model

def save_model(model, score, epoch=None, is_best=False, is_final=False, save_dir=None):
    """Save model checkpoint with minimal but essential metadata.
    
    Args:
        model: The PyTorch model to save
        score: Performance score of the model
        epoch: Current epoch number (optional)
        is_best: Whether this is the best model so far
        is_final: Whether this is the final model
        save_dir: Directory to save the model (defaults to current directory)
    """
    if save_dir is None:
        # Create default directory if none provided
        save_dir = os.path.join(os.getcwd(), "saved_models", 
                                f"nes_{datetime.now().strftime('%Y%m%d_%H%M')}")
        os.makedirs(save_dir, exist_ok=True)
    
    # Determine filename based on checkpoint type
    if is_best:
        filename = "best_model.pt"
    elif is_final:
        filename = "final_model.pt"
    else:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    # Save model with minimal metadata
    checkpoint = {
        'model_state': model.state_dict(),
        'score': score,
        'epoch': epoch,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path} (score: {score:.4f})")
    return save_path

if __name__ == "__main__":
    # Create output directory for saving models
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "saved_models", 
                          f"nes_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models will be saved to: {save_dir}")
    
    # Initialize a single model
    model = PKEModel(hidden_dim=HIDDEN_DIM)
    
    # Initialize with random parameters
    for name, param in model.named_parameters():
        nn.init.normal_(param, mean=0.0, std=0.02)
    
    print(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters")
    parameters = nn.utils.parameters_to_vector(model.parameters())
    print(f"Initial parameter norm: {parameters.norm().item():.4f}")

    # Training hyperparameters
    epochs = 50
    n_perturbations = mp.cpu_count() * 2
    if n_perturbations % 2 != 0: 
        n_perturbations += 1  # Ensure even number for antithetic sampling
    
    print(f"Using {n_perturbations} perturbations for NES")
    n_eval_runs = 4
    n_final_eval_runs = 16
    initial_perturb_size = 0.1  # Usually smaller in NES
    learning_rate = 0.01        # Learning rate for parameter updates
    perturb_decay = 0.995       # Slower decay for more stable convergence
    
    # Track model performance over time
    model_scores = []
    best_score = float('-inf')
    
    # Initial evaluation
    initial_score = evaluate_model_main(model, n_runs=n_eval_runs)
    model_scores.append(initial_score)
    print(f"Initial model score: {initial_score:.4f}")

    perturb_size = initial_perturb_size

    # Initialize multiprocessing pool
    num_workers = mp.cpu_count()
    print(f"Using {num_workers} worker processes.")
    pool = mp.Pool(processes=num_workers)

    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train model for one epoch using NES
            print(f"\nEpoch {epoch+1}/{epochs}")
            model = step(
                model, 
                n_perturb=n_perturbations, 
                perturb_size=perturb_size, 
                n_runs=n_eval_runs, 
                pool=pool, 
                learning_rate=learning_rate
            )
            
            # Evaluate model after training step
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                # More thorough evaluation at milestones
                epoch_score = evaluate_model_main(model, n_runs=n_eval_runs * 2)
            else:
                # Quick evaluation every epoch
                epoch_score = evaluate_model_main(model, n_runs=n_eval_runs)
                
            model_scores.append(epoch_score)
            print(f"Epoch {epoch+1} score: {epoch_score:.4f}")
            
            # Save model if it's the best so far or at regular intervals
            if epoch_score > best_score:
                best_score = epoch_score
                save_model(model, epoch_score, epoch, is_best=True, save_dir=save_dir)
            
            if (epoch + 1) % 10 == 0:  # Save checkpoint every 10 epochs
                save_model(model, epoch_score, epoch, save_dir=save_dir)
            
            # Decay perturbation size
            perturb_size *= perturb_decay
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_end_time - epoch_start_time:.2f}s")
            print(f"New perturbation size: {perturb_size:.4f}")

    finally:
        pool.close()
        pool.join()

    print("Training finished.")
    
    # Print score history
    print("\nScore history:")
    for epoch, score in enumerate(model_scores):
        print(f"  Epoch {epoch}: {score:.4f}")
    
    # Final evaluation
    final_score = evaluate_model_main(model, n_runs=n_final_eval_runs)
    print(f"Final model score after {epochs} epochs: {final_score:.4f}")
    print(f"Improvement: {(final_score - initial_score) / initial_score * 100:.2f}%")
    
    # Save final model
    save_model(model, final_score, epochs, is_final=True, save_dir=save_dir)
    
    print(f"Training finished. Best score: {best_score:.4f}, Final score: {final_score:.4f}")
    print(f"Improvement: {(final_score - initial_score) / initial_score * 100:.2f}%")