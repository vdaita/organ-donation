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


class PrioritySelectionPairedKidneyDonationEnv(PairedKidneyDonationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.MultiBinary(self.n_agents)

    def step(self, action, **kwargs):
        previous_matched = np.sum(self.matched_agents)
        selected_nodes = np.where(action == 1)[0]
        hard_nodes = np.where(self.is_hard_to_match[selected_nodes] == 1)[0]
        adj_matrix = rx.adjacency_matrix(self.current_graph)

        # print(adj_matrix, adj_matrix.shape)
        
        start_time = time.perf_counter()
        
        # create a new graph, give it hard nodes and selected nodes
        new_graph = rx.PyGraph()
        new_graph.add_nodes_from(selected_nodes)
        new_graph.add_nodes_from(hard_nodes)

        bidirectional_compatibility = np.logical_and(self.compatibility, self.compatibility.T)
        hard_and_unmatched = np.logical_and(self.is_hard_to_match, (self.matched_agents == 0))
        valid_edges = []
        for node in selected_nodes:
            mask = np.logical_and(adj_matrix[node], np.logical_and(bidirectional_compatibility[node], hard_and_unmatched))
            valid_neighbors = np.where(mask)[0]
            valid_edges.extend([(node, neighbor, 1) for neighbor in valid_neighbors])
        new_graph.add_edges_from(valid_edges)
        
        graph_const_time = time.perf_counter() - start_time
        
        # Find hard matching
        hard_matching_selected_nodes = rx.max_weight_matching(new_graph, max_cardinality=True)

        mwm_time = time.perf_counter() - start_time - graph_const_time
        
        # Process matched nodes
        nodes_to_remove = []
        for edge in hard_matching_selected_nodes:
            for node in edge:
                nodes_to_remove.append(node)
                self.node_matched(node)

        new_graph.remove_nodes_from(nodes_to_remove)
        new_graph.add_nodes_from(list(range(self.n_agents)))
        easy_edges = []
        
        unmatched = self.matched_agents == 0
        for node in selected_nodes:
            mask = np.logical_and(adj_matrix[node], np.logical_and(bidirectional_compatibility[node], unmatched))
            valid_neighbors = np.where(mask)[0]
            easy_edges.extend([(node, neighbor, 1) for neighbor in valid_neighbors])
        new_graph.add_edges_from(easy_edges)

        easy_graph_const_time = time.perf_counter() - start_time - graph_const_time - mwm_time
        
        # Find easy matching
        easy_matching_selected_nodes = rx.max_weight_matching(new_graph, max_cardinality=True)

        mwm_time2 = time.perf_counter() - start_time - graph_const_time - mwm_time - easy_graph_const_time

        for edge in easy_matching_selected_nodes:
            for node in edge:
                self.node_matched(node)

        final_time = time.perf_counter() - start_time - graph_const_time - mwm_time - mwm_time2 - easy_graph_const_time

        # print("Ratios: ")
        # print(f"Graph construction time: {graph_const_time:.4f} seconds")
        # print(f"Hard matching time: {mwm_time:.4f} seconds")
        # print(f"Easy graph construction time: {easy_graph_const_time:.4f} seconds")
        # print(f"Easy matching time: {mwm_time2:.4f} seconds")
        # print(f"Final time: {final_time:.4f} seconds")

        
        self.manage_arrivals_departures()
        self.current_step += 1
        done = self.current_step >= self.n_timesteps
        
        current_matched = np.sum(self.matched_agents)
        reward = (current_matched - previous_matched) / self.n_agents
        return self.get_observation(), reward, done, {}, {}
    

seed = 42
np.random.seed(seed)
random.seed(seed)

num_envs = 24
env_seeds = np.random.randint(0, 2**32 - 1, size=num_envs).tolist()

num_eval_envs = 256
eval_env_seeds = np.random.randint(0, 2**32 - 1, size=num_eval_envs).tolist()

n_agents = 100
n_timesteps = 32
death_time = 8
p = 0.037
q = 0.087
pct_hard = 0.6

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

greedy_rewards = []
for env in tqdm(envs, desc="Environments"):
    greedy_rewards.append(env.get_greedy_percentage())
greedy_rewards = np.array(greedy_rewards)
print(f"Greedy rewards: {greedy_rewards}")

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
    mean_ratios = np.mean(ratios)
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
    global greedy_rewards, envs

    solution = ga_instance.best_solution()[0]
    performance = evaluate_solution(solution)
    print("Evaluation descriptive stats: ")
    describe_performance(performance)

    # Reset the environment
    for env in envs:
        env.reset(seed=(env.seed + num_eval_envs))
    
    # Recalculate the greedy reward baselines
    greedy_rewards = []
    for env in tqdm(envs, desc="Environments"):
        greedy_rewards.append(env.get_greedy_percentage())
    greedy_rewards = np.array(greedy_rewards)

if __name__ == "__main__":
    sol_per_pop = 64
    num_genes = n_timesteps

    init_range_low = 0
    init_range_high = 2**6 - 1
    mutation_percent_genes = 25

    num_generations = 32
    num_parents_mating = sol_per_pop // 2

    # initial_population = np.ones((sol_per_pop, n_timesteps)) * (2**6 - 1) # select everything all the time (greedy)    
    
    random_population = np.random.randint(init_range_low, init_range_high + 1, size=(sol_per_pop // 2, n_timesteps))
    greedy_population = np.ones((sol_per_pop // 2, n_timesteps)) * (2**6 - 1) # select everything all the time (greedy)
    initial_population = np.vstack((random_population, greedy_population))

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
                        keep_elitism=5
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
