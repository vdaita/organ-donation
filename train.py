import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from model import DecisionTransformer
from gym_env import PairedOrganDonationEnv

def train(model, env, num_episodes=1000, lr=1e-4, gamma=0.99, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the model using simple policy gradient approach
    
    Args:
        model: DecisionTransformer model
        env: Gymnasium environment
        num_episodes: Number of episodes to train
        lr: Learning rate
        gamma: Discount factor
        device: Device to run on
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Track rewards
    episode_rewards = []
    
    # Create directories for saving results
    os.makedirs('results', exist_ok=True)
    
    for episode in tqdm(range(num_episodes)):
        # Reset environment
        observation, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Lists to store episode data
        log_probs = []
        rewards = []
        
        # Run episode
        while not done:
            # Get model input
            patients = torch.tensor(observation["patients"], dtype=torch.float32, device=device)
            matched_patients = torch.tensor(observation["matched_patients"], dtype=torch.bool, device=device)
            current_selection = torch.tensor(observation["current_selection"], dtype=torch.bool, device=device)
            
            # Forward pass to get action
            action = model(matched_patients, current_selection, patients)
            
            # Take step in environment
            next_observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store rewards
            rewards.append(reward)
            episode_reward += reward
            
            # Update observation
            observation = next_observation
        
        # Track episode rewards
        episode_rewards.append(episode_reward)
        
        # Only update model when episode has non-zero reward
        if sum(rewards) != 0:
            # Compute discounted rewards
            discounted_rewards = []
            running_reward = 0
            for r in reversed(rewards):
                running_reward = r + gamma * running_reward
                discounted_rewards.insert(0, running_reward)
            
            # Normalize rewards for stability
            discounted_rewards = torch.tensor(discounted_rewards, device=device)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
            
            # Compute loss and update
            optimizer.zero_grad()
            loss = -torch.sum(discounted_rewards)
            loss.backward()
            optimizer.step()
        
        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")
            
            # Save model checkpoint
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': avg_reward,
            }, f'results/model_checkpoint_{episode+1}.pt')
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes+1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.savefig('results/learning_curve.png')
    plt.close()
    
    # Save final model
    torch.save(model.state_dict(), 'results/final_model.pt')
    
    return episode_rewards

def evaluate(model, env, num_episodes=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the trained model
    
    Args:
        model: Trained DecisionTransformer model
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
        device: Device to run on
    """
    model = model.to(device)
    model.eval()
    
    eval_rewards = []
    
    for _ in tqdm(range(num_episodes)):
        observation, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get model input
            patients = torch.tensor(observation["patients"], dtype=torch.float32, device=device)
            matched_patients = torch.tensor(observation["matched_patients"], dtype=torch.bool, device=device)
            current_selection = torch.tensor(observation["current_selection"], dtype=torch.bool, device=device)
            
            # Forward pass to get action (without gradient tracking)
            with torch.no_grad():
                action = model(matched_patients, current_selection, patients)
            
            # Take step in environment
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    avg_reward = np.mean(eval_rewards)
    print(f"Evaluation over {num_episodes} episodes: Avg Reward = {avg_reward:.2f}")
    
    return eval_rewards

if __name__ == "__main__":
    # Initialize environment
    env = PairedOrganDonationEnv(num_pairs=16)
    
    # Get observation space info
    obs, _ = env.reset()
    n_attr = obs["patients"].shape[1]  # Number of attributes per patient
    
    # Initialize model
    model = DecisionTransformer(n_attr=n_attr, hidden_dim=64, n_heads=4, n_layers=2)
    
    print(f"Starting training with {model}")
    
    # Train model
    train_rewards = train(
        model=model,
        env=env,
        num_episodes=500,
        lr=1e-4
    )
    
    # Evaluate model
    eval_rewards = evaluate(model, env)
    
    print("Training and evaluation completed!")
