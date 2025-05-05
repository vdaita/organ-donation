# Instead of the patient strategy, which requires us to know when the patient is going to be leaving the simulation, we only have when the patient arrives, and whether or not they are hard to match. Then, we guess how long the patient stays and tries to match them with that.
from environment import PairedKidneyDonationEnv
import random
import numpy as np

# ONE OF THE CORE IDEAS THAT YOU'RE MISSING IS THAT THE SELECTED/IMPORTANT PATIENTS CAN MATCH WITH ANYONE!!! WE DON'T WANT TO RESTRICT THEM FROM PICKING UP ANYONE EXCEPT FOR PRIORITIZING THE HARD PEOPLE FIRST ADN THEN THE OTHERS LATER

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

envs = [
    PairedKidneyDonationEnv(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        death_time=death_time,
        seed=i,
        p=0.01,
        q=0.005,
        pct_hard=0.6
    )
    for i in seeds
]

for estimated_death_time in estimated_death_times:
    for env in envs:
        obs, _ = env.reset(seed=env.seed)
        done = False
        while not done:
            arrival_times = env["arrivals"]
            predicted_urgent = env["timestep"] >= arrival_times + estimated_death_time
            not_predicted_urgent = np.logical_not(predicted_urgent)
            
            action = obs["adjacency"]
            action[not_predicted_urgent, :] = 0

            obs, reward, done, _, _ = env.step(action >= 1)

