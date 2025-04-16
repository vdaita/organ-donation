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

class PPO:
    def __init__(self, num_layers=6, hidden_dim=32, device="cpu"):
        self.device = device

        self.actor_model = PairedKidneyModel(num_layers=num_layers, hidden_dim=hidden_dim)
        self.critic_model = PairedKidneyCriticModel(num_layers=num_layers, hidden_dim=hidden_dim)

        self.optimizer = torch.optim.Adam(
            list(self.actor_model.parameters()) + list(self.critic_model.parameters()),
            lr=0.001,
            eps=1e-5,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=16,
            gamma=0.1
        )

        self.discount_factor = 0.98
        self.gae_smoothing = 0.95
        self.clip_ratio = 0.2  # PPO clipping parameter

        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

    def sample_action(self, obs):
        probs = self.actor_model(obs).squeeze(-1)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        distribution = Bernoulli(probs=probs)
        action = distribution.sample()
        logprob = distribution.log_prob(action).sum()
        return action, logprob, probs

    def get_action(self, obs):
        probs = self.actor_model(obs).squeeze(-1)
        # Convert probabilities to binary actions with threshold of 0.5
        return (probs >= 0.5).float()

    def compute_values_for_episode(self, env: PairedKidneyDonationEnv):
        self.actor_model.train()
        self.critic_model.train()

        obs, info = env.start_over()
        done = False

        observations = []
        actions = []
        rewards = []
        values = []
        logprobs = []
        probs_list = []

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
            observations.append(obs)
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
            
        return observations, actions, returns, advantages, logprobs, probs_list

    def update(self, observations, actions, returns, advantages, old_logprobs, old_probs):
        # PPO update with clipping
        policy_loss = 0
        value_loss = 0
        
        for obs, action, ret, adv, old_logprob, old_prob in zip(observations, actions, returns, advantages, old_logprobs, old_probs):
            if obs["active_agents"].sum() == 0:
                continue
                
            # Get new action probabilities and value estimate
            new_probs = self.actor_model(obs).squeeze(-1)
            new_probs = torch.clamp(new_probs, 1e-6, 1.0 - 1e-6)
            distribution = Bernoulli(probs=new_probs)
            new_logprob = distribution.log_prob(action).sum()
            value = self.critic_model(obs)
            
            # Calculate ratio for PPO clipping
            ratio = torch.exp(new_logprob - old_logprob)
            
            # Clipped surrogate objective
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            policy_loss += -torch.min(ratio * adv, clip_adv)
            
            # Value loss
            value_loss += (ret - value) ** 2
            
        policy_loss /= max(1, len(observations))
        value_loss /= max(1, len(observations))
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(list(self.actor_model.parameters()) + list(self.critic_model.parameters()), 0.5)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()

# training constants
episodes_per_env = 256
eval_every = 16
num_eval_runs = 4
    
agent = PPO(
    device="cpu", # faster on CPU - probably because less data transferring back and forth w environment
)

envs = [
    PairedKidneyDonationEnv(
        n_agents=1000,
        n_timesteps=36,
        criticality_rate=18
    )
]

def eval_model(env, agent, num_runs):
    agent.actor_model.eval()
    agent.critic_model.eval()
    
    eval_model_rewards = []
    eval_greedy_rewards = []
    eval_patient_rewards = []

    for i in tqdm(range(num_runs)):
        obs, info = env.reset()
        done = False
        reward = 0  # Initialize reward
        while not done:
            # Get binary actions from the model
            action = agent.get_action(obs)
            obs, reward, done, _, info = env.step(action)
        eval_model_rewards.append(reward)
        eval_greedy_rewards.append(get_greedy_percentage(env))
        eval_patient_rewards.append(get_patient_percentage(env))

    return {
        "model": eval_model_rewards,
        "greedy": eval_greedy_rewards,
        "patient": eval_patient_rewards,
    }

for env in envs:
    for episode in tqdm(range(episodes_per_env)):
        env.reset()
        observations, actions, returns, advantages, logprobs, probs_list = agent.compute_values_for_episode(env)
        policy_loss, value_loss = agent.update(observations, actions, returns, advantages, logprobs, probs_list)

        if (episode + 1) % agent.scheduler.step_size == 0:
            agent.scheduler.step()

        if (episode + 1) % eval_every == 0:
            evaluations = eval_model(env, agent, num_runs=num_eval_runs)
            mean_eval_greedy_reward = np.mean(evaluations["greedy"])
            mean_eval_patient_reward = np.mean(evaluations["patient"])
            mean_eval_model_reward = np.mean(evaluations["model"])

            print(f"Evaluation {episode + 1}: "
                f"avg reward = {mean_eval_model_reward:.2f}, "
                f"greedy = {mean_eval_greedy_reward:.2f}, "
                f"patient = {mean_eval_patient_reward:.2f}")
        
    with open("model_results.json", "w+") as f: # the values we are writing here should be from the last round
        f.write(json.dumps({
            "greedy": evaluations["greedy"],
            "patient": evaluations["patient"],
            "model": evaluations["model"]
        }))


    env.close()

torch.save(agent.actor_model.state_dict(), "actor_model.pth")
torch.save(agent.critic_model.state_dict(), "critic_model.pth")