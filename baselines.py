from environment import PairedKidneyDonationEnv
from gymnasium.spaces import Dict
import numpy as np
import time

# Greedy solution
env = PairedKidneyDonationEnv(n_agents=2000, n_timesteps=360, criticality_rate=50)
env.reset()
done = False

print("Optimal situation (in retrospect): ", env.get_theoretical_max())

global_start_time = time.time()
while not done:
    start_time = time.time()
    action = {
        "selection": np.zeros(env.n_agents),
        "match_selection": 0,
        "match_regular": 1
    }
    observation, reward, done, info = env.step(action)
    end_time = time.time()
global_end_time = time.time()

print("-> Greedy solution")
print("Time taken: ", global_end_time - global_start_time, " Last reward: ", reward, " Hard to match rate: ", info["hard_to_match_rate"], " Easy to match rate: ", info["regular_match_rate"])

# Time period solution
