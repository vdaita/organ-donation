from environment import PairedKidneyDonationEnv
from gymnasium.spaces import Dict
import numpy as np
import time
import gymnasium as gym
from tqdm import tqdm
import json
from model import PairedKidneyModel
import torch
from copy import deepcopy

def get_greedy_percentage(env: PairedKidneyDonationEnv): # easier to test with PairedKidneyDonationEnv directly
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        action = np.ones(env.n_agents)
        observation, new_reward, done, _, info = env.step(action)
        reward += new_reward
    return reward

def get_periodic_percentage(env: PairedKidneyDonationEnv, period_timesteps: int):
    done = False
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        if env.current_step % period_timesteps == 0:
            action = np.ones(env.n_agents)
            obs, new_reward, done, _, info = env.step(action)
        else:
            action = np.zeros(env.n_agents)
            obs, new_reward, done, _,  info = env.step(action)
        reward += new_reward
    return reward

def get_periodic_greedy_mixed(env: PairedKidneyDonationEnv, period_timesteps: int):
    done = False
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        if env.current_step % period_timesteps == 0:
            action = np.ones(env.n_agents)
            observation, new_reward, done, _, info = env.step(action)
        else:
            action = env.is_hard_to_match
            observation, new_reward, done, _, info = env.step(action)
        reward += new_reward
    return reward   

def get_patient_percentage(env: PairedKidneyDonationEnv):
    obs, info = env.start_over()
    reward, done = 0, False
    while not done:
        # check if any elements are just before the timestep when they depart?
        selection = np.zeros(env.n_agents)
        for i in range(env.n_agents):
            if env.real_departure_times[i] - env.current_step == 1:
                selection[i] = 1
        if env.current_step == env.n_timesteps - 1:
            selection = np.ones(env.n_agents)
    
        action = selection
        observation, new_reward, done, _, info = env.step(action)
        reward += new_reward
    return reward   

def get_greedy_patient_mixed(env: PairedKidneyDonationEnv):
    obs, info = env.start_over(seed=env.seed)
    reward, done = 0, False
    while not done:
        # check if there are any elements that are just before the timestep when they depart? or they are hard to match?
        selection = np.zeros(env.n_agents)
        for i in range(env.n_agents):
            if env.real_departure_times[i] - env.current_step == 1 or env.is_hard_to_match[i] == 1:
                selection[i] = 1
        action = selection
        observation, new_reward, done, _, info = env.step(action)
        reward += new_reward
    return reward

if __name__ == "__main__":
    periods = [2, 4, 7] # when 180 timesteps and 90 criticality rate, each day represents 4 days
    # twice a week, once every 2 weeks, once a month (roughly)
    num_simulations = 16
    num_agents = [500, 1000, 2000]
    use_model = False # TODO: fix! doesn't work

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

            simulation_result["results"].append({
                "type": {
                    "method": "patient"
                },
                "reward": get_patient_percentage(env)
            })

            simulation_result["results"].append({
                "type": {
                    "method": "greedy-patient-mixed"
                },
                "reward": get_greedy_patient_mixed(env)
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

            if use_model:
                model = PairedKidneyModel(
                    hidden_dim=32,
                    num_layers=8
                )
                model.load_state_dict(torch.load("model.pth"))

                obs, info = env.start_over()
                done = False
                while not done:
                    tensor_state = deepcopy(obs)
                    for key in tensor_state:
                        if not (isinstance(obs[key], int) or isinstance(obs[key], float)):
                            tensor_state[key] = torch.Tensor(obs[key])

                    action = model(
                        tensor_state["adjacency_matrix"], 
                        obs["timestep"], 
                        tensor_state["arrivals"], 
                        tensor_state["departures"],
                        tensor_state["is_hard_to_match"], 
                        obs["total_timesteps"], 
                        tensor_state["active_agents"]
                    )
                    action = action.cpu().detach().numpy()
                    action = {
                        "selection": action,
                        "match_selection": 1,
                        "match_regular": 0
                    }
                    obs, reward, done, _, info = env.step(action)
                
                simulation_result["results"].append({
                    "type": {
                        "method": "model"
                    },
                    "reward": reward
                })

            simulation_results.append(simulation_result)    

    with open("results/simulation_results.json", "w") as f:
        json.dump(simulation_results, f, indent=4)        
