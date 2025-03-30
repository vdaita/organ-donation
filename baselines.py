from environment import PairedKidneyDonationEnv
from gymnasium.spaces import Dict
import numpy as np
import time

def get_greedy_percentage(env: PairedKidneyDonationEnv):
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        action = {
            "selection": np.zeros(env.n_agents),
            "match_selection": 0,
            "match_regular": 1
        }
        observation, reward, done, info = env.step(action)

    env.start_over()
    return reward

def get_periodic_percentage(env: PairedKidneyDonationEnv, period_timesteps: int):
    done = False
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        if env.current_timestep % period_timesteps == 0:
            action = {
                "selection": np.zeros(env.n_agents),
                "match_selection": 0,
                "match_regular": 1
            }
            observation, reward, done, info = env.step(action)
        else:
            action = {
                "selection": np.zeros(env.n_agents),
                "match_selection": 0,
                "match_regular": 1
            }
            observation, reward, done, info = env.step(action)
    env.start_over()
    return reward

if __name__ == "__main__":
    env = PairedKidneyDonationEnv(
        n_agents=2000,
        n_timesteps=360,
        criticality_rate=180
    )
    greedy_reward = get_greedy_percentage(env)
    print(f"Greedy reward: {greedy_reward}")

    periodic_reward = get_periodic_percentage(env, period_timesteps=30)
    print(f"Periodic reward: {periodic_reward}")