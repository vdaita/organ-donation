# If you thicken the market with patients who are less sick, how will the patients that are sick do? Will they get more matches?
import numpy as np
from environment import PrioritySelectionPairedKidneyDonationEnv
from gymnasium.spaces import Dict, MultiBinary
import matplotlib.pyplot as plt
from tqdm import tqdm

num_envs = 128
n_agents_original = 300
p = 0.037
q = 0.087
pct_hard = 0.6
death_time = 32
n_timesteps = 64

class LessSickPairedKidneyDonationEnv(PrioritySelectionPairedKidneyDonationEnv):
    def __init__(self, pct_less_sick=0.7, **kwargs):
        self.pct_less_sick = pct_less_sick
        super().__init__(**kwargs)
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        count_less_sick = int(self.n_agents * self.pct_less_sick)
        less_sick_indices = self.np_random.choice(self.n_agents, size=count_less_sick, replace=False)
        self.is_less_sick = np.zeros(self.n_agents, dtype=np.int8)
        self.is_less_sick[less_sick_indices] = 1
        self.real_departure_times[less_sick_indices] = self.n_timesteps
        
        return obs, info

    def get_observation(self):
        obs = super().get_observation()
        return obs

    def get_greedy_percentage(self, with_prioritization=False):
        obs, _ = self.start_over()
        done = False
        while not done:
            action = np.ones(self.n_agents)
            if with_prioritization:
                action[np.where(self.is_less_sick == 1)] = 0
            obs, reward, done, _, _ = self.step(action)
        total_reward = np.sum(self.matched_agents) / self.n_agents
        return total_reward, self.get_sick_matched()

    def get_sick_matched(self):
        sick_indices = np.where(self.is_less_sick == 0)[0]
        if len(sick_indices) == 0:
            return 0
        return np.sum(self.matched_agents[sick_indices]) / len(sick_indices)

if __name__ == "__main__":
    regular_envs = [
        PrioritySelectionPairedKidneyDonationEnv(
            n_agents=n_agents_original,
            n_timesteps=n_timesteps,
            death_time=death_time,
            seed=i,
            p=p,
            q=q,
            pct_hard=pct_hard,
        )
        for i in range(num_envs)
    ]

    stats = {
        "regular": []
    }
    
    pct_less_sick_values = [0.05, 0.1, 0.25, 0.33, 0.5]

    for pct_less_sick in tqdm(pct_less_sick_values, desc="Less sick envs"):
        new_agents_count = int(n_agents_original * (1 / (1 - pct_less_sick)))
        less_sick_envs = [
            LessSickPairedKidneyDonationEnv(
                n_agents=new_agents_count,
                n_timesteps=n_timesteps,
                death_time=death_time,
                seed=i,
                p=p,
                q=q,
                pct_hard=pct_hard,
                pct_less_sick=pct_less_sick
            )
            for i in range(num_envs)
        ]

        stats[f"{pct_less_sick}x_less_sick_with_prioritization"] = []
        for env in tqdm(less_sick_envs, desc="Environments", leave=False):
            total_reward, sick_matched = env.get_greedy_percentage(with_prioritization=True)
            stats[f"{pct_less_sick}x_less_sick_with_prioritization"].append(sick_matched)

        stats[f"{pct_less_sick}x_less_sick_without_prioritization"] = []
        for env in tqdm(less_sick_envs, desc="Environments", leave=False):
            total_reward, sick_matched = env.get_greedy_percentage(with_prioritization=False)
            stats[f"{pct_less_sick}x_less_sick_without_prioritization"].append(sick_matched)
    
    for env in tqdm(regular_envs, desc="Regular envs"):
        stats["regular"].append(env.get_greedy_percentage())

    labels = []
    data = []

    # Collect data for regular environment
    labels.append("regular")
    data.append(stats["regular"])

    # Collect data for each less sick percentage
    for pct_less_sick in pct_less_sick_values:
        key_no_prio = f"{pct_less_sick}x_less_sick_without_prioritization"
        key_prio = f"{pct_less_sick}x_less_sick_with_prioritization"
        labels.append(f"{int(pct_less_sick*100)}% less sick\nno prioritization")
        data.append(stats[key_no_prio])
        labels.append(f"{int(pct_less_sick*100)}% less sick\nwith prioritization")
        data.append(stats[key_prio])

    plt.figure(figsize=(14, 7))
    box = plt.boxplot(
        data, 
        labels=labels, 
        patch_artist=True, 
        widths=0.5, 
        showfliers=False
    )

    # Color alternate boxes for clarity
    colors = ['#8ecae6', '#219ebc'] * (len(labels)//2 + 1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel("Fraction of Sick Patients Matched")
    plt.title("Sick Patients Matched vs. Fraction of Less Sick Patients in Market")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout(pad=2)
    plt.subplots_adjust(bottom=0.18)
    plt.savefig("results/less_sick_patients.png", dpi=300)
    plt.show()