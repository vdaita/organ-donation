import gymnasium as gym
from tqdm import tqdm
import numpy as np
from environment import PairedKidneyDonationEnv
from scipy.stats import gmean
import pygad

num_envs = 32
env_seeds = [i for i in range(0, num_envs)]

n_agents = 100
n_timesteps = 64
death_range = [28, 32]

scores = {}

envs = [
    PairedKidneyDonationEnv(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        death_range=death_range,
        seed=i,
        p=0.01,
        q=0.001
    )
    for i in env_seeds
]

greedy_rewards = []

for env in tqdm(envs, desc="Environments"):
    greedy_rewards.append(env.get_greedy_percentage())
greedy_rewards = np.array(greedy_rewards)
print(f"Greedy rewards: {greedy_rewards}")

def translate_number_to_action(number, obs): # number should be from 0 to 2^6 - 1
    bits = f"{number:06b}"
    selection = np.array([[int(bits[0]), int(bits[1]), int(bits[2])],
                       [int(bits[3]), int(bits[4]), int(bits[5])]])
    
    timestep = obs["timestep"]

    N, _ = obs["adjacency"].shape
    is_hard = obs["is_hard"]
    is_early = (timestep - obs["arrivals"]) <= 1 # 0 and 1
    is_late = (obs["departures"] - timestep) <= 2
    is_middle = np.logical_not(np.logical_or(is_early, is_late))
    times = np.zeros(N, dtype=int)
    times[is_early] = 0
    times[is_middle] = 1
    times[is_late] = 2

    action = obs["adjacency"].copy()

    for htm_index in range(2): # easy to match, hard to match
        for time_index in range(3): # early, middle, late
            if selection[htm_index, time_index] == 0: # if you should not select this, then zero out the action
                relevant_nodes = np.where(np.logical_and(is_hard == htm_index, times == time_index))[0]
                action[relevant_nodes, :] = 0
                action[:, relevant_nodes] = 0

    return action

def play_schedule_game(ga_instance, schedule, solution_idx):
    env = PairedKidneyDonationEnv(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        death_range=death_range
    )
    model_rewards = []
    for env in tqdm(envs, desc="Environments", leave=False):
        obs, _ = env.start_over()
        done = False
        while not done:
            action = translate_number_to_action(schedule[obs["timestep"]], obs)
            obs, reward, done, _, _ = env.step(action)
        model_reward = np.sum(env.matched_agents) / env.n_agents
        model_rewards.append(model_reward)
    model_rewards = np.array(model_rewards)
    ratios = model_rewards / greedy_rewards
    # print(f"Model rewards: {model_rewards}")
    # print(f"Ratios: {ratios}")
    mean_ratios = np.mean(ratios)
    scores[solution_idx] = ratios

    # if gmean_ratios > 1:
    #     print("-> Better than greedy")
    #     print(f"Ratios: {ratios}")
    #     print(f"Schedule: {schedule}")

    return mean_ratios

def on_fitness(ga_instance, population_fitness):
    print("Fitness of the population: ", population_fitness)

if __name__ == "__main__":
    sol_per_pop = 16
    num_genes = n_timesteps
    init_range_low = 0
    init_range_high = 2**6 - 1
    mutation_percent_genes = 20

    num_generations = 32
    num_parents_mating = sol_per_pop // 2

    # initial_population = np.ones((sol_per_pop, n_timesteps)) * (2**6 - 1) # select everything all the time (greedy)
    best_last = [40, 58, 6, 3, 45, 18, 18, 63, 48, 63, 37, 37, 5, 11, 50, 35, 0, 24, 49, 33, 63, 60, 60, 3, 
             13, 63, 63, 14, 63, 2, 44, 10, 63, 60, 9, 35, 58, 11, 12, 27, 19, 47, 49, 27, 44, 23, 63, 49,
             49, 23, 28, 63, 52, 49, 0, 26, 63, 63, 51, 25, 31, 52, 22, 53]
    initial_population = np.array([best_last] * sol_per_pop)
    

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=play_schedule_game,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       on_fitness=on_fitness,
                       initial_population=initial_population,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_type=int,
                       gene_space=list(range(init_range_low, init_range_high + 1))
                    )
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    print(f"Scores: {scores[solution_idx]}")