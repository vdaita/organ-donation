# If you thicken the market with patients who are less sick, how will the patients that are sick do? Will they get more matches?
import numpy as np
from environment import PrioritySelectionPairedKidneyDonationEnv

num_envs = 128
n_agents_original = 300
p = 0.037
q = 0.087
pct_hard = 0.6
death_time = 32
n_timesteps = 64


class LessSickPairedKidneyDonationEnv(PrioritySelectionPairedKidneyDonationEnv):
    def __init__(self, pct_less_sick = 0.7, **kwargs):
        super().__init__(**kwargs)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        count_less_sick = int(self.n_agents * self.pct_less_sick)
        less_sick_indices = self.np_random.choice(self.n_agents, size=count_less_sick, replace=False)
        self.is_less_sick = np.zeros(self.n_agents, dtype=bool)
        self.is_less_sick[less_sick_indices] = True
        self.real_departure_times[less_sick_indices] = self.n_timesteps

    def get_greedy_percentage(self):
        obs, _ = self.start_over()
        done = False
        while not done:
            action = np.ones(self.n_agents)
            action[self.is_less_sick] = 0
            obs, reward, done, _, _ = self.step(action)
        total_reward = np.sum(self.matched_agents) / self.n_agents
        return total_reward

    def get_sick_matched(self):
        return np.sum(self.matched_agents[~self.is_less_sick]) / np.sum(~self.is_less_sick)
    
if __name__ == "__main__":
    less_sick_envs = [
        LessSickPairedKidneyDonationEnv(
            n_agents=n_agents_original * (1 / (1 - pct_less_sick)),
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

    sick_matched = {
        "regular": [regular_e]
    }

    for pct_less_sick in [0.25, 0.33, 0.5, 0.75]:
