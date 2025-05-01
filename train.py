# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from model import PairedKidneyCriticModel, PairedKidneyModel
from environment import PairedKidneyDonationEnv
import copy
from baselines import get_greedy_percentage

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-5
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.3
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    num_runs_eval: int = 16

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # environment configurations
    n_agents = 100
    n_timesteps = 32
    death_low = 12
    death_high = 14
    use_cycles = False
    p = 0.08
    q = 0.04
    pct_hard = 0.7

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = PairedKidneyCriticModel(hidden_dim=64)
        self.actor = PairedKidneyModel(hidden_dim=64)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = self.actor(x)
        # Reshape probs to match action space
        if len(probs.shape) == 3:  # [B, N, N]
            batch_size, n_nodes = probs.shape[0], probs.shape[1]
        else:  # [N, N]
            batch_size, n_nodes = 1, probs.shape[0]
            probs = probs.unsqueeze(0)
            
        # Create Bernoulli distribution for each edge
        std = 0.2
        dist = Normal(probs, std)
        
        if action is None:
            action = dist.sample()

        action = torch.clamp(action, -0.999, 0.999)
            
        # Calculate log probabilities for the entire matrix
        log_prob = dist.log_prob(action)
        
        # Get adjacency matrices to mask log probs
        device = action.device
        
        # Handle adjacency matrix extraction properly
        if isinstance(x, dict):
            # Single observation or dictionary of batched observations
            if isinstance(x["adjacency"], list):
                # Handle batched  in dictionary format
                adj_matrix = torch.tensor(np.array(x["adjacency"])).to(device)
            else:
                # Handle single observation
                adj_matrix = torch.tensor(x["adjacency"]).to(device)
                if len(adj_matrix.shape) == 2:
                    adj_matrix = adj_matrix.unsqueeze(0)
        else:
            # List of observation dictionaries
            adj_matrix = torch.stack([torch.tensor(obs["adjacency"]).to(device) for obs in x])
        
        # Make sure adjacency matrix has same shape as log_prob
        if adj_matrix.shape != log_prob.shape:
            if len(adj_matrix.shape) < len(log_prob.shape):
                # Add batch dimension if needed
                adj_matrix = adj_matrix.unsqueeze(0)
            
        # Only count log probs for valid edges
        masked_log_prob = log_prob * adj_matrix
        logprob = masked_log_prob.sum(dim=(1, 2))
        
        # Same for entropy
        entropy = dist.entropy() * adj_matrix
        entropy_sum = entropy.sum(dim=(1, 2))
        
        return action, logprob, entropy_sum, self.critic(x)
    
    def evaluate_model(self, step, env, num_runs=16, logger=None):
        self.actor.eval()
        self.critic.eval()

        model_percentages = []
        greedy_percentages = []

        for i in range(num_runs):
            env.reset()
            greedy_percentage = env.get_greedy_percentage()
            greedy_percentages.append(greedy_percentage)

            obs, info = env.start_over()
            env.matched_agents = np.zeros_like(env.matched_agents)
            done = False
            total_reward = 0
            while not done:
                with torch.no_grad():
                    action = self.actor(obs)
                obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                total_reward += reward
            model_percentages.append(sum(env.matched_agents) / len(env.matched_agents))

        ratios = [model_pct / greedy_pct for model_pct, greedy_pct in zip(model_percentages, greedy_percentages)]
        print(
            "Model reward: ", model_percentages,
            "\nGreedy reward: ", greedy_percentages,
            "\nRatios: ", ratios,
            "\nMean ratio: ", np.mean(ratios),
            "\nStd ratio: ", np.std(ratios),
        )

        if logger is not None:
            run.track(np.mean(ratios), name="reward_ratio", step=step)

        self.actor.train()
        self.critic.train()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        from aim import Run
        run = Run()
        run["hparams"] = vars(args)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    def generate_env():
        return PairedKidneyDonationEnv(
            n_agents=args.n_agents,
            n_timesteps=args.n_timesteps,
            death_range=[args.death_low, args.death_high],
            use_cycles=args.use_cycles,
            p=args.p,
            q=args.q,
            pct_hard=args.pct_hard,
        )

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [lambda: generate_env() for _ in range(args.num_envs)],
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent().to(device)
    agent.actor.reset_parameters()
    agent.critic.reset_parameters()

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = [None] * args.num_steps
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # print(next_obs)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # print("Action: ", action.shape, "Logprob: ", logprob.shape, "Value: ", value.shape)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = next_obs, torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_logprobs = logprobs.reshape(-1)
        # print("Old logprobs shape: ", logprobs.shape, "New logprobs shape: ", b_logprobs.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # print("Old actions shape: ", actions.shape, "New actions shape: ", b_actions.shape)
        # print("Old advantages shape: ", advantages.shape, "New advantages shape: ", b_advantages.shape)
        # print("Old returns shape: ", returns.shape, "New returns shape: ", b_returns.shape)
        # print("Old values shape: ", values.shape, "New values shape: ", b_values.shape)

        b_obs = {}
        for key in obs[0].keys():
            stacked_key_data = torch.stack([torch.as_tensor(o[key], device=device) for o in obs], dim=0)
            num_steps, num_envs = stacked_key_data.shape[0], stacked_key_data.shape[1]
            b_obs[key] = stacked_key_data.reshape(num_steps * num_envs, *stacked_key_data.shape[2:])

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {key: val[mb_inds] for key, val in b_obs.items()}
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time))
        }
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step)
            if args.track:
                run.track(value, name=key, step=global_step)

        agent.evaluate_model(global_step, generate_env(), num_runs=args.num_runs_eval, logger=run if args.track else None)
        print("SPS: ", metrics["charts/SPS"])

    envs.close()
    writer.close()