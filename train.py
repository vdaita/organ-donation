from environment import PairedKidneyDonationEnv
from model import PairedKidneyModel, PairedKidneyCriticModel
import numpy as np
from torch.distributions.bernoulli import Bernoulli
import torch
import gymnasium as gym
from tqdm import tqdm
from baselines import get_greedy_percentage, get_periodic_percentage, get_patient_percentage
import json
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch_geometric.nn import GAT, global_mean_pool
from copy import deepcopy
from aim import Run
import time
import os

lr = 1e-3
weight_decay = 1e-5
num_layers = 3
hidden_dim = 32

class PPO:
    def __init__(self, num_layers=6, hidden_dim=32, device="cpu"):
        self.device = device

        self.actor_model = PairedKidneyModel(num_layers=num_layers, hidden_dim=hidden_dim)
        self.critic_model = PairedKidneyCriticModel(num_layers=num_layers, hidden_dim=hidden_dim)


        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(),
            lr=lr,
            eps=1e-8,
            weight_decay=weight_decay
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(),
            lr=lr * 3, # apparently this helps the critic converge faster which should help the actor in turn
            eps=1e-8,
            weight_decay=weight_decay
        )

        self.discount_factor = 0.98
        self.gae_smoothing = 0.9
        self.clip_ratio = 0.2  # PPO clipping parameter
        self.max_grad_norm = 0.5  # For gradient clipping

        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

    def sample_action(self, obs):
        probs = self.actor_model(obs).squeeze(-1)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        # Check for NaN values and replace with safe defaults
        mask = torch.isnan(probs)
        if mask.any():
            probs = torch.where(mask, torch.ones_like(probs) * 1e-6, probs)
        distribution = Bernoulli(probs=probs)
        action = distribution.sample()
        logprob = distribution.log_prob(action).sum()
        return action, logprob, probs

    def get_action(self, obs):
        probs = self.actor_model(obs).squeeze(-1)
        # Handle NaN values that might appear during inference
        mask = torch.isnan(probs)
        if mask.any():
            probs = torch.where(mask, torch.zeros_like(probs), probs)
        return (probs >= 0.5).float()

    def compute_values_for_episode(self, env: PairedKidneyDonationEnv):
        self.actor_model.train()
        self.critic_model.train()

        obs, info = env.reset()
        done = False

        observations = []
        actions = []
        rewards = []
        values = []
        logprobs = []
        probs_list = []

        with torch.no_grad():
            while not done:
                if obs["active_agents"].sum() == 0:
                    action = torch.zeros((obs["active_agents"].shape[0],), device=self.device)
                    logprob = torch.tensor(0.0, device=self.device)
                    probs = torch.zeros((obs["active_agents"].shape[0],), device=self.device)
                else:
                    action, logprob, probs = self.sample_action(obs)

                # Get value estimate
                value = self.critic_model(obs) if obs["active_agents"].sum() > 0 else torch.tensor(0.0, device=self.device)
                
                # Store current state info                
                observations.append(deepcopy(obs))
                actions.append(action)
                values.append(value)
                logprobs.append(logprob)
                probs_list.append(probs)
                
                # Take a step in the environment
                obs, reward, done, _, info = env.step(action)
                reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
                rewards.append(reward)
            
        # Stack tensors
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        values = torch.stack(values)
        logprobs = torch.stack(logprobs)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # No next state at the end of episode
            else:
                next_value = values[t + 1].item()
                
            delta = rewards[t].item() + self.discount_factor * next_value - values[t].item()
            gae = delta + self.discount_factor * self.gae_smoothing * gae
            returns.insert(0, gae + values[t].item())
            advantages.insert(0, gae)
            
        returns = torch.tensor(returns, device=self.device)
        advantages = torch.tensor(advantages, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        num_valid_observation = sum([1 for obs in observations if obs["active_agents"].sum() > 0])            
        return observations, actions, returns, advantages, logprobs, values

    def update(self, observations, actions, returns, advantages, old_logprobs, values, num_rounds=3):
        valid_indices = [i for i, obs in enumerate(observations) if obs["active_agents"].sum() > 0]
        
        if not valid_indices:
            return 0, 0  # No valid observations to update on
        
        observations = [observations[i] for i in valid_indices]
        actions = actions[valid_indices]
        returns = returns[valid_indices]
        advantages = advantages[valid_indices]
        old_logprobs = old_logprobs[valid_indices]
        values = values[valid_indices]
        
        policy_loss_sum = 0
        value_loss_sum = 0
        
        for _ in range(num_rounds):
            # Calculate new log probabilities and value predictions
            new_logprobs_list = []
            new_values_list = []
            entropies = []
            
            for i, obs in enumerate(observations):
                # Get updated policy probabilities
                probs = self.actor_model(obs).squeeze(-1)
                probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
                
                # Check for NaN values and replace with safe defaults
                mask = torch.isnan(probs)
                if mask.any():
                    probs = torch.where(mask, torch.ones_like(probs) * 1e-6, probs)
                
                # Calculate log probabilities for the actions we took
                distribution = Bernoulli(probs=probs)
                new_logprob = distribution.log_prob(actions[i]).sum()
                entropy = distribution.entropy().mean()
                entropies.append(entropy)
                new_logprobs_list.append(new_logprob)
                
                # Get updated value estimate
                new_value = self.critic_model(obs)
                new_values_list.append(new_value)
            
            # Stack results
            new_logprobs = torch.stack(new_logprobs_list)
            new_values = torch.stack(new_values_list)
            
            # Calculate ratios for PPO clipping
            ratios = torch.exp(new_logprobs - old_logprobs)
            # Clip ratios to prevent numerical instability 
            ratios = torch.clamp(ratios, 0.0, 10.0)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Entropy bonus
            entropy_bonus = torch.stack(entropies).mean()
            
            # Calculate policy loss (negative for gradient ascent)
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_bonus
            
            # Calculate value function loss
            value_loss = 0.5 * ((new_values.squeeze() - returns) ** 2).mean()

            # Perform backpropagation and optimization
            self.actor_optimizer.zero_grad()
            policy_loss.backward()

            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.actor_model.parameters(),
                self.max_grad_norm
            )
            self.actor_optimizer.step()

            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic_model.parameters(),
                self.max_grad_norm
            )

            self.critic_optimizer.step()
            
            # Track losses
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
        
        # Return average losses
        return policy_loss_sum / num_rounds, value_loss_sum / num_rounds

# training constants
episodes_per_env = 1024
eval_every = 32
batch_size = 4
num_eval_runs = 16
use_cycles = True
    
agent = PPO(
    device="cpu", # faster on CPU - probably because less data transferring back and forth w environment
    num_layers=num_layers,
    hidden_dim=hidden_dim
)

envs = [
    PairedKidneyDonationEnv(
        n_agents=500,
        n_timesteps=64,
        criticality_rate=8,
        use_cycles=use_cycles,
    )
]

run = Run()
run["hparams"] = {
    "learning_rate": lr,
    "env_agents": envs[0].n_agents,
    "env_timesteps": envs[0].n_timesteps,
    "env_criticality_rate": envs[0].criticality_rate,
    "episodes_per_env": episodes_per_env,
    "batch_size": batch_size,
    "num_layers": num_layers,
    "hidden_dim": hidden_dim,
    "use_cycles": use_cycles
}

def eval_model(env, agent, num_runs):
    agent.actor_model.eval()
    agent.critic_model.eval()
    
    eval_model_rewards = []
    eval_greedy_rewards = []
    eval_patient_rewards = []

    waiting_times = []

    for i in tqdm(range(num_runs)):
        run_waiting_times = {}

        obs, info = env.reset(options={"should_log": False})
        done = False
        reward = 0  # Initialize reward
        while not done:
            # Get binary actions from the model
            action = agent.get_action(obs)
            obs, new_reward, done, _, info = env.step(action)
            reward += new_reward # accumulate reward
        eval_model_rewards.append(reward)
        run_waiting_times["model"] = env.get_waiting_time_stats()
        eval_greedy_rewards.append(get_greedy_percentage(env))
        run_waiting_times["greedy"] = env.get_waiting_time_stats()
        eval_patient_rewards.append(get_patient_percentage(env))
        run_waiting_times["patient"] = env.get_waiting_time_stats()

        if i == num_runs - 1:
            print("-> Environment info:")
            env.print_info(env.get_info())
            print("-> Model waiting time stats:")
            env.print_waiting_time_stats(run_waiting_times["model"])
            print("-> Greedy waiting time stats:")
            env.print_waiting_time_stats(run_waiting_times["greedy"])
            print("-> Patient waiting time stats:")
            env.print_waiting_time_stats(run_waiting_times["patient"])

        waiting_times.append(run_waiting_times)

    return {
        "model": eval_model_rewards,
        "greedy": eval_greedy_rewards,
        "patient": eval_patient_rewards,
        "waiting_times": waiting_times
    }

timestamp = time.time()
os.makedirs(f"results/{timestamp}", exist_ok=True)

for env in envs:
    for episode in tqdm(range(episodes_per_env)):
        observations, actions, returns, advantages, logprobs, values = agent.compute_values_for_episode(env)
        policy_loss, value_loss = agent.update(observations, actions, returns, advantages, logprobs, values)
        run.track(policy_loss, name="policy_loss", step=episode)
        run.track(value_loss, name="value_loss", step=episode)

        if (episode + 1) % eval_every == 0:
            evaluations = eval_model(env, agent, num_runs=num_eval_runs)
            mean_eval_greedy_reward = np.mean(evaluations["greedy"])
            mean_eval_patient_reward = np.mean(evaluations["patient"])
            mean_eval_model_reward = np.mean(evaluations["model"])

            print(f"Evaluation {episode + 1}: "
                f"avg reward = {mean_eval_model_reward:.2f}, "
                f"greedy = {mean_eval_greedy_reward:.2f}, "
                f"patient = {mean_eval_patient_reward:.2f}")
            
            model_greedy_ratio = mean_eval_model_reward / mean_eval_greedy_reward
            model_patient_ratio = mean_eval_model_reward / mean_eval_patient_reward

            run.track(model_greedy_ratio, name="model_greedy_ratio", step=episode)
            run.track(model_patient_ratio, name="model_patient_ratio", step=episode)

        
    with open(f"results/{timestamp}/model_results.json", "w+") as f: # the values we are writing here should be from the last round
        f.write(json.dumps({
            "greedy": evaluations["greedy"],
            "patient": evaluations["patient"],
            "model": evaluations["model"]
        }))


    env.close()

torch.save(agent.actor_model.state_dict(), f"results/{timestamp}/actor_model.pth")
torch.save(agent.critic_model.state_dict(), f"results/{timestamp}/critic_model.pth")