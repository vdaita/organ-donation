import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Bernoulli
import gymnasium as gym
from environment import PairedKidneyDonationEnv
from simple_model import SimpleEdgePredictor, SimpleCritic

# Configuration
CONFIG = {
    'lr': 3e-4,
    'n_agents': 100,
    'n_timesteps': 64,
    'hidden_dim': 64,
    'n_episodes': 1000,
    'gamma': 0.99,
    'ppo_epochs': 3,
    'clip_ratio': 0.2,
    'batch_size': 16,
    'p': 0.02,
    'q': 0.05,
    'pct_hard': 0.7
}

def evaluate_model(actor, env, n_episodes=10):
    """Evaluate the current model"""
    actor.eval()
    returns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action_probs = actor(obs)
                action = (action_probs > 0.5).float().cpu().numpy()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        returns.append(episode_reward)
        
    actor.train()
    return np.mean(returns)

def main():
    # Create environment
    env = PairedKidneyDonationEnv(
        n_agents=CONFIG['n_agents'],
        n_timesteps=CONFIG['n_timesteps'],
        p=CONFIG['p'],
        q=CONFIG['q'],
        pct_hard=CONFIG['pct_hard']
    )
    
    # Create actor-critic models
    actor = SimpleEdgePredictor(CONFIG['hidden_dim'])
    critic = SimpleCritic(CONFIG['hidden_dim'])
    
    # Set up optimizer
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), 
        lr=CONFIG['lr']
    )
    
    # Training loop
    best_return = -float('inf')
    
    for episode in range(CONFIG['n_episodes']):
        # Collect trajectory
        obs, _ = env.reset()
        obs_list, action_list, reward_list, value_list, logprob_list = [], [], [], [], []
        done = False
        episode_reward = 0
        
        while not done:
            # Get action probabilities and value
            with torch.no_grad():
                action_probs = actor(obs)
                value = critic(obs)
                
                # Sample action using Bernoulli
                dist = Bernoulli(probs=action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Mask log probs with adjacency matrix
                adj_matrix = torch.FloatTensor(obs["adjacency_matrix"])
                masked_log_prob = log_prob * adj_matrix
                logprob = masked_log_prob.sum()
            
            # Execute action in environment
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            value_list.append(value)
            logprob_list.append(logprob)
            
            obs = next_obs
        
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for r, v in zip(reversed(reward_list), reversed(value_list)):
            R = r + CONFIG['gamma'] * R
            advantage = R - v
            returns.insert(0, R)
            advantages.insert(0, advantage)
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy using PPO
        for _ in range(CONFIG['ppo_epochs']):
            for idx in range(0, len(obs_list), CONFIG['batch_size']):
                batch_obs = obs_list[idx:idx + CONFIG['batch_size']]
                batch_actions = action_list[idx:idx + CONFIG['batch_size']]
                batch_advantages = advantages[idx:idx + CONFIG['batch_size']]
                batch_returns = returns[idx:idx + CONFIG['batch_size']]
                batch_logprobs = logprob_list[idx:idx + CONFIG['batch_size']]
                
                # Get new action probabilities and values
                new_values = []
                new_logprobs = []
                
                for o, a in zip(batch_obs, batch_actions):
                    action_probs = actor(o)
                    value = critic(o)
                    new_values.append(value)
                    
                    dist = Bernoulli(probs=action_probs)
                    log_prob = dist.log_prob(a)
                    
                    # Mask log probs with adjacency matrix
                    adj_matrix = torch.FloatTensor(o["adjacency_matrix"])
                    masked_log_prob = log_prob * adj_matrix
                    new_logprobs.append(masked_log_prob.sum())
                
                new_values = torch.stack(new_values)
                new_logprobs = torch.stack(new_logprobs)
                
                # Calculate ratios and PPO loss
                ratios = torch.exp(new_logprobs - torch.stack(batch_logprobs))
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-CONFIG['clip_ratio'], 1+CONFIG['clip_ratio']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = ((new_values - batch_returns) ** 2).mean()
                
                # Combined loss
                loss = policy_loss + 0.5 * value_loss
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate and print progress
        if episode % 10 == 0:
            eval_return = evaluate_model(actor, env)
            print(f"Episode {episode}, Return: {episode_reward:.2f}, Eval Return: {eval_return:.2f}")
            
            if eval_return > best_return:
                best_return = eval_return
                torch.save(actor.state_dict(), "best_actor.pt")
                torch.save(critic.state_dict(), "best_critic.pt")
    
    print(f"Training complete. Best return: {best_return:.2f}")

if __name__ == "__main__":
    main()
