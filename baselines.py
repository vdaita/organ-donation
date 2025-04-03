from environment import PairedKidneyDonationEnv
from gymnasium.spaces import Dict
import numpy as np
import time
import gymnasium as gym
from tqdm import tqdm
import json

def get_greedy_percentage(env: PairedKidneyDonationEnv): # easier to test with PairedKidneyDonationEnv directly
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        action = {
            "selection": np.zeros(env.n_agents),
            "match_selection": 0,
            "match_regular": 1
        }
        observation, reward, done, _, info = env.step(action)

    env.start_over()
    return reward

def get_periodic_percentage(env: PairedKidneyDonationEnv, period_timesteps: int):
    done = False
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        if env.current_step % period_timesteps == 0:
            action = {
                "selection": np.zeros(env.n_agents),
                "match_selection": 0,
                "match_regular": 1
            }
            obs, reward, done, _, info = env.step(action)
        else:
            action = {
                "selection": np.zeros(env.n_agents),
                "match_selection": 0,
                "match_regular": 0
            }
            obs, reward, done, _,  info = env.step(action)
    env.start_over()
    return reward

def get_periodic_greedy_mixed(env: PairedKidneyDonationEnv, period_timesteps: int):
    done = False
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        if env.current_step % period_timesteps == 0:
            action = {
                "selection": np.zeros(env.n_agents),
                "match_selection": 0,
                "match_regular": 1
            }
            observation, reward, done, _, info = env.step(action)
        else:
            action = {
                "selection": env.is_hard_to_match,
                "match_selection": 1,
                "match_regular": 0
            }
            observation, reward, done, _, info = env.step(action)
    env.start_over()
    return reward            

if __name__ == "__main__":
    periods = [2, 4, 7] # when 180 timesteps and 90 criticality rate, each day represents 4 days
    # twice a week, once every 2 weeks, once a month (roughly)
    num_simulations = 100
    num_agents = [500, 1000, 2000]

    simulation_results = []

    for agent_count in tqdm(num_agents, desc="Agents", leave=False):
        env = PairedKidneyDonationEnv(
            n_agents=agent_count,
            n_timesteps=180,
            criticality_rate=90
        )
        for simulation in tqdm(range(num_simulations), desc="Simulations", leave=False):
            env.reset(seed=simulation)

            greedy_reward = get_greedy_percentage(env)
            simulation_result = {
                "num_agents": agent_count,
                "simulation_number": simulation, 
                "results": []
            }

            simulation_result["results"].append({
                "type": {
                    "method": "greedy"
                },
                "reward": greedy_reward
            })

            for period in tqdm(periods, desc="Periods", leave=False):
                periodic_reward = get_periodic_percentage(env, period)
                mixed_reward = get_periodic_greedy_mixed(env, period)
                simulation_result["results"].append({
                    "type": {
                        "method": "periodic",
                        "period": period
                    },
                    "reward": periodic_reward
                })
                simulation_result["results"].append({
                    "type": {
                        "method": "mixed",
                        "period": period
                    },
                    "reward": mixed_reward
                })

            simulation_results.append(simulation_result)    

    with open("simulation_results.json", "w") as f:
        json.dump(simulation_results, f, indent=4)        
