import gymnasium as gym
from tqdm import tqdm
import numpy as np
from environment import PairedKidneyDonationEnv
from scipy.stats import gmean
import pygad
import random
import matplotlib.pyplot as plt
import rustworkx as rx
import multiprocessing as mp
from copy import deepcopy
import gymnasium as gym
import time
import numba as nb

@nb.njit(fastmath=True, parallel=True)
def get_edges(selected_nodes, matchable_nodes, adj_matrix):
    adj_bidirectional = np.logical_and(adj_matrix, adj_matrix.T)
    adj_view = adj_bidirectional[selected_nodes][:, matchable_nodes]
    edges = np.argwhere(adj_view == 1)
    edges = [(selected_nodes[edge[0]], matchable_nodes[edge[1]], 1) for edge in edges]
    return edges

class PrioritySelectionPairedKidneyDonationEnv(PairedKidneyDonationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.MultiBinary(self.n_agents)

    def match_subgraph(self, selected_nodes, matchable_nodes, adj_matrix):
        start_time = time.perf_counter()
        graph = rx.PyGraph()
        graph.add_nodes_from(range(self.n_agents))

        adj_matrix_time = time.perf_counter() - start_time
        
        rx_edges = get_edges(selected_nodes, matchable_nodes, adj_matrix)
        graph.add_edges_from(rx_edges)

        edge_add_time = time.perf_counter() - start_time - adj_matrix_time

        matching = rx.max_weight_matching(graph, max_cardinality=True)

        matching_time = time.perf_counter() - start_time - edge_add_time - adj_matrix_time

        for pair in matching:
            self.node_matched(pair[0])
            self.node_matched(pair[1])

        end_time = time.perf_counter() - start_time - adj_matrix_time - edge_add_time - matching_time

        # print(f"Adjacency matrix time: {adj_matrix_time:.6f} seconds")
        # print(f"Edge addition time: {edge_add_time:.6f} seconds")
        # print(f"Matching time: {matching_time:.6f} seconds")
        # print(f"Marking time: {end_time:.6f} seconds")
       

        # print(f"Matching time: {end_time - start_time:.6f} seconds")


    def step(self, action, **kwargs):
        previous_matched = np.sum(self.matched_agents)

        selected_nodes = np.where(action == 1)[0]
        hard_indices = np.where(self.is_hard_to_match == 1)[0]
        
        adj_matrix = rx.adjacency_matrix(self.current_graph)
        
        self.match_subgraph(selected_nodes, hard_indices, adj_matrix)
        self.match_subgraph(selected_nodes, np.arange(self.n_agents), adj_matrix)

        self.manage_arrivals_departures()
        self.current_step += 1
        done = self.current_step >= self.n_timesteps
        
        current_matched = np.sum(self.matched_agents)
        reward = (current_matched - previous_matched) / self.n_agents
        return self.get_observation(), reward, done, {}, {}
    

    def get_greedy_percentage(self):
        obs, _ = self.start_over()
        done = False
        while not done:
            action = np.ones(self.n_agents)
            obs, reward, done, _, _ = self.step(action)
        total_reward = np.sum(self.matched_agents) / self.n_agents
        return total_reward

    def get_patient_percentage(self):
        obs, _ = self.start_over()
        done = False
        while not done:
            action = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                if self.real_departure_times[i] - self.current_step == 1:
                    action[i] = 1
            obs, reward, done, _, _ = self.step(action)
        total_reward = np.sum(self.matched_agents) / self.n_agents
        return total_reward

seed = 42
np.random.seed(seed)
random.seed(seed)

num_envs = 16
env_seeds = np.random.randint(0, 2**32 - 1, size=num_envs).tolist()

num_eval_envs = 128
eval_env_seeds = np.random.randint(0, 2**32 - 1, size=num_eval_envs).tolist()

n_agents = 100
n_timesteps = 32
death_time = 16
p = 0.037 * 2
q = 0.087 * 2
pct_hard = 0.7

envs = [
    PrioritySelectionPairedKidneyDonationEnv(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        death_time=death_time,
        seed=i,
        p=p,
        q=q,
        pct_hard=pct_hard
    )
    for i in env_seeds
]

eval_envs = [
    PrioritySelectionPairedKidneyDonationEnv(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        death_time=death_time,
        seed=i,
        p=p,
        q=q,
        pct_hard=pct_hard
    )
    for i in eval_env_seeds
]


eval_greedy_rewards = []
for env in tqdm(eval_envs, desc="Evaluation envs"):
    eval_greedy_rewards.append(env.get_greedy_percentage())
eval_greedy_rewards = np.array(eval_greedy_rewards)

def translate_number_to_action(number, obs): # number should be from 0 to 2^6 - 1
    bits = f"{number:06b}"
    selection = np.array([[int(bits[0]), int(bits[1]), int(bits[2])],
                       [int(bits[3]), int(bits[4]), int(bits[5])]])
    
    timestep = obs["timestep"]

    N, _ = obs["adjacency"].shape
    is_hard = obs["is_hard"]

    time_remaining = obs["departures"] - timestep
    is_urgent = time_remaining <= 1
    is_soon = (time_remaining > 1) & (time_remaining <= 3)
    is_early = time_remaining > 3
    
    times = np.zeros(N, dtype=int)
    times[is_early] = 0
    times[is_soon] = 1
    times[is_urgent] = 2

    action = obs["adjacency"].copy()

    for htm_index in range(2): # easy to match, hard to match
        for time_index in range(3): # early, middle, late
            if selection[htm_index, time_index] == 0: # if you should not select this, then zero out the action
                relevant_nodes = np.where(np.logical_and(is_hard == htm_index, times == time_index))[0]
                action[relevant_nodes, :] = 0
                action[:, relevant_nodes] = 0

    return action

def adapt_generated_schedule(schedule, n_timesteps):
    schedule_len = len(schedule)
    if n_timesteps % schedule_len != 0:
        raise ValueError("Schedule length must be a divisor of n_timesteps.")
    return np.array([schedule[i % schedule_len] for i in range(n_timesteps)])

def evaluate_env(env, schedule) -> float:

    obs, _  = env.start_over()
    done = False
    while not done:
        action = translate_number_to_action(schedule[obs["timestep"]], obs)
        obs, reward, done, _, _ = env.step(action)
    total_reward = np.sum(env.matched_agents) / env.n_agents
    return total_reward

def play_schedule_game(ga_instance, schedule, solution_idx):
    model_rewards = []
    for env in tqdm(envs, desc="Environments", leave=False):
        model_rewards.append(evaluate_env(env, schedule))
    model_rewards = np.array(model_rewards)

    ratios = model_rewards / greedy_rewards

    failures = np.sum(ratios < 1)
    failure_ratio = failures / len(ratios)
   
    if failure_ratio > 0.3:
        return np.mean(ratios) * 0.5

    ratios[ratios < 1] = ratios[ratios < 1] ** 2
    mean_ratios = gmean(ratios)
    first_tenth = np.percentile(ratios, 10)
    if first_tenth < 1:
        mean_ratios = (mean_ratios + 2 * first_tenth) / 3

    return mean_ratios

def on_fitness(ga_instance, population_fitness):
    print("Fitness of the population: ", population_fitness)

def evaluate_solution(schedule):
    model_rewards = []
    for env in tqdm(eval_envs, desc="Environments", leave=False):
        model_rewards.append(evaluate_env(env, schedule))
    model_rewards = np.array(model_rewards)

    ratios = model_rewards / eval_greedy_rewards
    return ratios

def describe_performance(performance):
    print(f"Mean: {np.mean(performance)}")
    print(f"Std: {np.std(performance)}")
    print(f"Min: {np.min(performance)}")
    print(f"Max: {np.max(performance)}")
    print(f"Geometric mean: {gmean(performance)}")
    print(f"Median: {np.median(performance)}")
    print(f"10th percentile: {np.percentile(performance, 10)}")
    print(f"25th percentile: {np.percentile(performance, 25)}")
    print(f"75th percentile: {np.percentile(performance, 75)}")
    print(f"90th percentile: {np.percentile(performance, 90)}")

def on_generation(ga_instance):
    global greedy_rewards, num_envs, envs

    solution = ga_instance.best_solution()[0]
    print("Solution: ", solution)

    performance = evaluate_solution(solution)
    print("Evaluation descriptive stats: ")
    describe_performance(performance)

    reset_environments()

def reset_environments():
    global greedy_rewards, num_envs, envs, theoretical_rewards, patient_rewads

    random_seeds = np.random.randint(0, 2**32 - 1, size=len(envs)).tolist()
    envs = [
        PrioritySelectionPairedKidneyDonationEnv(
            n_agents=n_agents,
            n_timesteps=n_timesteps,
            death_time=death_time,
            seed=i,
            p=p,
            q=q,
            pct_hard=pct_hard
        )
        for i in random_seeds
    ]
    
    greedy_rewards = []
    for env in tqdm(envs, desc="Environments"):
        greedy_rewards.append(env.get_greedy_percentage())
    greedy_rewards = np.array(greedy_rewards)
    print(f"Greedy rewards: {greedy_rewards}")

    theoretical_rewards = []
    for env in tqdm(envs, desc="Environments"):
        theoretical_rewards.append(env.calculate_theoretical_max())
    theoretical_rewards = np.array(theoretical_rewards)
    print(f"Theoretical rewards: {theoretical_rewards}")

    patient_rewards = []
    for env in tqdm(envs, desc="Environments"):
        patient_rewards.append(env.get_patient_percentage())
    patient_rewards = np.array(patient_rewards)
    print(f"Patient rewards: {patient_rewards}")

if __name__ == "__main__":
    sol_per_pop = 24
    num_genes = n_timesteps

    init_range_low = 0
    init_range_high = 2**6 - 1
    mutation_percent_genes = 10

    num_generations = 16
    num_parents_mating = sol_per_pop // 2

    # initial_population = np.ones((sol_per_pop, n_timesteps)) * (2**6 - 1) # select everything all the time (greedy)    
    
    greedy_population = np.ones((sol_per_pop // 4, n_timesteps)) * (2**6 - 1) # select everything all the time (greedy)
    
    batched_population_4 = np.zeros((sol_per_pop // 4, n_timesteps))
    for i in range(0, n_timesteps, 4):
        batched_population_4[:, i] = (2 ** 6 - 1)

    batched_population_6 = np.zeros((sol_per_pop // 4, n_timesteps))
    for i in range(0, n_timesteps, 6):
        batched_population_6[:, i] = (2 ** 6 - 1)

    patient_population = np.ones((sol_per_pop // 4, n_timesteps)) * (2 ** 2 + 2**5)

    initial_population = np.vstack((greedy_population, batched_population_4, batched_population_6, patient_population))

    reset_environments()

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
                        gene_space=list(range(init_range_low, init_range_high + 1)),
                        on_generation=on_generation,
                    )
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    eval_ratios = evaluate_solution(solution)

    print("Evaluation stats: ")
    describe_performance(eval_ratios)

    print(f"Final eval ratios: {eval_ratios}")
    plt.title("Ratios")
    plt.boxplot(eval_ratios)
    plt.savefig("results/schedule_ratios.png")
    plt.show()
