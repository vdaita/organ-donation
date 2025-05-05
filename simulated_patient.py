# Instead of the patient strategy, which requires us to know when the patient is going to be leaving the simulation, we only have when the patient arrives, and whether or not they are hard to match. Then, we guess how long the patient stays and tries to match them with that.
from environment import PairedKidneyDonationEnv
import random
import numpy as np

seed = 42
np.random.seed(seed)
random.seed(seed)

n_agents = 100
n_timesteps = 32
death_time = 16

num_envs = 128

estimated_death_times = [
    ((int) (death_time * 0.5),), 
    ((int) (death_time * 0.75),), 
    ((int) (death_time * 0.9),)
]

seeds = np.random(0, 2**32 - 1)

env = PairedKidneyDonationEnv(
    n_agents=n_agents,
    n_timesteps=n_timesteps,
    death_time=death_time,
    seed=seed,
    p=0.037,
    q=0.087
)

for estimated_death_time in estimated_death_times:
