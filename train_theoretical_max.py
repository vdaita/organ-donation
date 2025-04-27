"""
Currently, my approaches so far have been using reinforcement learning. What if the theoretical maximum is used instead?
"""
from binary_decision_environment import BinaryDecisionEnvironment
from torch import nn
from tqdm import tqdm
import torch
import numpy as np
from aim import Run

model = nn.Sequential(
    nn.Linear(12, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
# randomly initialize the model
model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)


lr = 0.0005
weight_decay = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

if __name__ == "__main__":
    num_environments = 2048
    batch_size = 64
    envs_per_eval = 64 * 8
    runs_per_eval = 8
    n_agents = 100

    env = BinaryDecisionEnvironment(n_agents=n_agents)
    run = Run()
    run["hparams"] = {
        "num_environments": num_environments,
        "batch_size": batch_size,
        "envs_per_eval": envs_per_eval,
        "runs_per_eval": runs_per_eval,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_agents": n_agents
    }

    step = 0

    for environment_idx in tqdm(range(num_environments)):
        env.reset(seed=environment_idx)
        best_matching, all_edges = env.get_theoretical_max()
        bad_matching = [edge for edge in all_edges if edge not in best_matching]

        matchings = best_matching + bad_matching
        scores = ([1] * len(best_matching)) + ([0] * len(bad_matching))

        index_reshuffle = np.random.permutation(len(matchings))

        num_matchings = len(matchings)
        pos_weight = len(bad_matching) / len(best_matching)
        
        outputs = torch.zeros(num_matchings, 1)
        for batch in range(0, num_matchings, batch_size):
            batch_elements = index_reshuffle[batch:batch + batch_size]

            batch_matchings = [matchings[i] for i in batch_elements]
            batch_outputs = torch.zeros(len(batch_matchings))
            batch_target_scores = torch.tensor([scores[i] for i in batch_elements], dtype=torch.float32)

            optimizer.zero_grad()

            for match_idx, matching in enumerate(batch_matchings):
                a, b = matching
                start = max(env.arrival_times[a], env.arrival_times[b])
                end = min(env.departure_times[a], env.departure_times[b])

                features_list = [env._edge_to_feature(matching, current_timestep=timestep) for timestep in range(start, end + 1)]
                features = np.array(features_list)
                features = torch.tensor(features, dtype=torch.float32)

                outputs = model(features)
                batch_outputs[match_idx] = torch.max(outputs)
                # max_output = torch.max(outputs)

                # if max_output > 0.5:
                #     for output in outputs:
                #         if output > 0.5:
                #             batch_outputs[match_idx] = output
                # else:
                #     batch_outputs[match_idx] = torch.mean(outputs)
    
            loss_fn = nn.BCELoss(
                weight=torch.tensor([1.0 if score == 1 else (1 / pos_weight) for score in batch_target_scores]),
                reduction="mean"
            )
            loss = loss_fn(batch_outputs, batch_target_scores)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Calculate weighted accuracy
            predictions = (batch_outputs > 0.5).float()
            correct = (predictions == batch_target_scores).float()
            weights = batch_target_scores * pos_weight + (1 - batch_target_scores)
            weighted_acc = torch.sum(correct * weights) / torch.sum(weights)

            run.track(weighted_acc.item(), name="weighted_accuracy", step=step)
        
            step += 1
            run.track(loss.item(), name="loss", step=step)

        if environment_idx % envs_per_eval == 0:
            model_rewards = []
            greedy_rewards = []
            theoretical_max_rewards = []

            for env_idx in range(runs_per_eval):
                obs, _ = env.reset(seed=environment_idx * env_idx)
                done = False
                while not done:
                    action = model(torch.tensor(obs, dtype=torch.float32))
                    obs, reward, done, _, _ = env.step(action > 0.5)
                model_rewards.append(reward)
                greedy_rewards.append(env.get_greedy_result())
                theoretical_max_rewards.append((len(env.get_theoretical_max()[0]) * 2) / n_agents)

            print("Model rewards: ", np.mean(model_rewards)) 
            print("Greedy rewards: ", np.mean(greedy_rewards))
            print("Theoretical max rewards: ", np.mean(theoretical_max_rewards))
            
            ratio = [model_reward / greedy_reward for model_reward, greedy_reward in zip(model_rewards, greedy_rewards)]
            print("Ratio: ", np.mean(ratio), " +/- ", np.std(ratio))

            run.track(np.mean(ratio), name="ratio", step=step)