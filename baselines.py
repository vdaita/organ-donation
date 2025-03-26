from environment import PairedKidneyDonationEnv
from gymnasium.spaces import Dict
import numpy as np

# Greedy solution
env = PairedKidneyDonationEnv()
env.reset()
done = False

print("Optimal situation (in retrospect): ", env.get_theoretical_max())

while not done:
    action = {
        "selection": np.zeros(env.n_agents),
        "match_selection": 0,
        "match_regular": 1
    }
    observation, reward, done, info = env.step(action)
    print(reward)