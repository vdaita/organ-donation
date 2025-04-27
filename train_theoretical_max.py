"""
Currently, my approaches so far have been using reinforcement learning. What if the theoretical maximum is used instead?
"""
from binary_decision_environment import BinaryDecisionEnvironment
from torch import nn
from tqdm import tqdm
import torch

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# randomly initialize the model
model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if __name__ == "__main__":
    num_environments = 2096
    batch_size = 32
    envs_per_eval = 32
    runs_per_eval = 8

    env = BinaryDecisionEnvironment(n_agents=250)

    for environment_idx in tqdm(range(num_environments)):
        env.reset()
        best_matching, all_edges = env.get_theoretical_max()
        bad_matching = [edge for edge in all_edges if edge not in best_matching]

        matchings = best_matching + bad_matching
        scores = ([1] * len(best_matching)) + ([0] * len(bad_matching))

        num_matchings = len(matchings)
        
        outputs = torch.zeros(num_matchings, 1)
        for batch in range(0, num_matchings, batch_size):
            batch_min = batch
            batch_max = min(batch + batch_size, num_matchings)

            batch_matchings = matchings[batch_min:batch_max]
            batch_outputs = torch.zeros(len(batch_matchings), 1)
            batch_target_scores = torch.tensor(scores[batch_min:batch_max], dtype=torch.float32)

            for match_idx, matching in enumerate(batch_matchings):
                a, b = matching
                start = max(env.arrival_times[a], env.arrival_times[b])
                end = min(env.departure_times[a], env.departure_times[b])

                features = [env._edge_to_feature(matching, current_timestep=timestep) for timestep in range(start, end + 1)]
                features = torch.tensor(features, dtype=torch.float32)

                outputs = model(features)
                batch_outputs[match_idx] = torch.max(outputs)
            
            loss = nn.MSELoss()(batch_outputs, batch_target_scores)
            loss.backward()
            optimizer.step()

        if environment_idx % envs_per_eval == 0:
            model_rewards = []
            greedy_rewards = []

            for _ in range(runs_per_eval):
                env.reset()
                obs, _ = env.reset(seed=environment_idx)
                done = False
                while not done:
                    action, _ = model(torch.tensor(obs, dtype=torch.float32))
                    obs, reward, done, _, _ = env.step(action)
                model_rewards.append(reward)
                greedy_rewards.append(env.get_greedy_result())
            
            print("Model rewards: ", model_rewards) 
            print("Greedy rewards: ", greedy_rewards)
            ratio = [model_reward / greedy_reward for model_reward, greedy_reward in zip(model_rewards, greedy_rewards)]
            print("Ratio: ", ratio)