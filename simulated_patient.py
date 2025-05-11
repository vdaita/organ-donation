# Instead of the patient strategy, which requires us to know when the patient is going to be leaving the simulation, we only have when the patient arrives, and whether or not they are hard to match. Then, we guess how long the patient stays and tries to match them with that.
from schedule_game import PrioritySelectionPairedKidneyDonationEnv
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    n_agents = 400
    n_timesteps = 128
    death_time = 64

    num_envs = 16
    p = 0.037
    q = 0.087
    pct_hard = 0.6

    estimated_death_times = {
        "0.5 * patient": ((int) (death_time * 0.5),),
        "0.75 * patient": ((int) (death_time * 0.75),),
        "0.9 * patient": ((int) (death_time * 0.9),)
    }

    seeds = np.random.randint(0, 2 ** 32 - 1, size=(num_envs, )).tolist()

    envs = [
        PrioritySelectionPairedKidneyDonationEnv(
            n_agents=n_agents,
            n_timesteps=n_timesteps,
            death_time=death_time,
            seed=i,
            p=p,
            q=q,
            pct_hard=pct_hard
        )
        for i in seeds
    ]

    print("Finished constructing environments")

    greedy_rewards = []
    for env in tqdm(envs, desc="Greedy: Environments"):
        greedy_rewards.append(env.get_greedy_percentage())
    greedy_rewards = np.array(greedy_rewards)

    patient_rewards = []
    for env in tqdm(envs, desc="Patient: Environments"):
        patient_rewards.append(env.get_patient_percentage())
    patient_rewards = np.array(patient_rewards)

    ratios = {
        "patient": patient_rewards / greedy_rewards,
        "greedy": greedy_rewards / greedy_rewards
    }

    for strategy_name in tqdm(estimated_death_times, desc="Strategies"):
        estimated_death_time = estimated_death_times[strategy_name]
        strategy_rewards = []
        for env in tqdm(envs, desc="Environments", leave=False):
            obs, _ = env.reset(seed=env.seed)
            done = False
            while not done:
                arrival_times = obs["arrivals"]
                predicted_urgent = obs["timestep"] >= arrival_times + estimated_death_time - 1
                obs, reward, done, _, _ = env.step(predicted_urgent)
            strategy_rewards.append(np.sum(env.matched_agents) / env.n_agents)
        strategy_rewards = np.array(strategy_rewards)
        ratios[strategy_name] = strategy_rewards / greedy_rewards

    plt.figure(figsize=(10, 6))
    labels = list(ratios.keys())
    data = [ratios[label] for label in labels]

    print("Greedy strategy cost: ")
    print(np.mean(greedy_rewards))
    print("Patient strategy cost: ")
    print(np.mean(patient_rewards))


    plt.boxplot(data, labels=labels)
    plt.ylabel("Reward Ratio")
    plt.title("Reward Ratios by Strategy")
    plt.grid(axis='y')
    plt.savefig("results/simulated_patient.png")
    plt.show()