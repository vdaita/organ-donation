import numpy as np
import networkx as nx
from gymnasium.spaces import Graph, MultiBinary, Dict, Box, Discrete, MultiDiscrete
import time
import random
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
random.seed(seed)

ttn = {
    1: "O",
    2: "A",
    3: "B",
    4: "AB"
}
ntt = { v: k for k, v in ttn.items() }

blood_type_distribution = { # borrowed from Ergin, Sonmez, and Umver
    1: 0.3,
    2: 0.4,
    3: 0.2,
    4: 0.1
}
can_give = {
    1: [1, 3],
    2: [2, 3],
    3: [3],
    4: [1, 2, 3, 4]
}

n_agents = 50

def get_random_blood_type():
    return np.random.choice(list(blood_type_distribution.keys()), p=list(blood_type_distribution.values()))

def create_environment(n_agents):
    triplets = []
    for i in range(n_agents):
        p = get_random_blood_type()
        d1 = get_random_blood_type()
        d2 = get_random_blood_type()
        triplets.append((p, d1, d2))
    return triplets

def is_grouping_viable(grouping):
    donors = [groups[1] for groups in grouping] + [group[2] for group in grouping]
    recipients = [group[0] for group in grouping]
    donors, recipients = np.array(donors), np.array(recipients)

    a_count = np.sum(donors == ntt["A"])
    b_count = np.sum(donors == ntt["B"])
    ab_count = np.sum(donors == ntt["AB"])
    o_count = np.sum(donors == ntt["O"])

    donor_counts = np.array([0, a_count, b_count, ab_count, o_count])
    
    # check o type
    donor_counts[ntt["O"]] -= np.sum(recipients == ntt["O"])
    if donor_counts[ntt["O"]] < 0:
        return False

    # Remove A-type recipients: first from A donors, then from O donors
    a_needed = np.sum(recipients == ntt["A"])
    take_from_a = min(donor_counts[ntt["A"]], a_needed)
    donor_counts[ntt["A"]] -= take_from_a
    a_needed -= take_from_a
    donor_counts[ntt["O"]] -= a_needed
    if donor_counts[ntt["O"]] < 0:
        return False

    # Remove B-type recipients: first from B donors, then from O donors
    b_needed = np.sum(recipients == ntt["B"])
    take_from_b = min(donor_counts[ntt["B"]], b_needed)
    donor_counts[ntt["B"]] -= take_from_b
    b_needed -= take_from_b
    donor_counts[ntt["O"]] -= b_needed
    if donor_counts[ntt["O"]] < 0:
        return False
    
    # Remove AB-type recipients: first from AB donors, then from A and B donors, then from O donors
    ab_needed = np.sum(recipients == ntt["AB"])
    take_from_ab = min(donor_counts[ntt["AB"]], ab_needed)
    donor_counts[ntt["AB"]] -= take_from_ab
    ab_needed -= take_from_ab
    take_from_a = min(donor_counts[ntt["A"]], ab_needed)
    donor_counts[ntt["A"]] -= take_from_a
    ab_needed -= take_from_a
    take_from_b = min(donor_counts[ntt["B"]], ab_needed)
    donor_counts[ntt["B"]] -= take_from_b
    ab_needed -= take_from_b
    donor_counts[ntt["O"]] -= ab_needed
    if donor_counts[ntt["O"]] < 0:
        return False
    
    return True

def find_valid_groupings(triplets):
    # produce a list of all possible groupings of 2/3 elements
    pairs = []
    for i in range(len(triplets)):
        for j in range(i + 1, len(triplets)):
            if not i == j:
                grouping = [triplets[i], triplets[j]]
                if is_grouping_viable(grouping):
                    pairs.append(grouping)
    triples = []
    for i in range(len(triplets)):
        for j in range(i + 1, len(triplets)):
            for k in range(j + 1, len(triplets)):
                if not i == j and not i == k and not j == k:
                    # check if the triplet is viable
                    grouping = [triplets[i], triplets[j], triplets[k]]
                    if is_grouping_viable(grouping):
                        triples.append(grouping)

    groupings = pairs + triples
    return groupings

def permute_grouping(solution, num_changes):
    new_solution = solution.copy()

    for _ in range(num_changes):
        swap_i = np.random.randint(0, len(solution))
        swap_j = np.random.randint(0, len(solution))
        new_solution[swap_i], new_solution[swap_j] = new_solution[swap_j], new_solution[swap_i]

    return new_solution

def toggle_elements(solution, num_changes):
    # select num_changes indices to toggle out of len(solution)
    indices = np.random.choice(len(solution), num_changes, replace=False)
    new_solution = solution.copy()
    new_solution[indices] = 1 - new_solution[indices]
    return new_solution

def score_permutation(solution, groupings):
    sorted_groupings = [groupings[i] for i in solution]  # sorted groupings
    matched = set()
    total_triplets_matched = 0
    
    for group in sorted_groupings:
        # Check if any triplet in this group is already matched
        group_valid = True
        for triplet in group:
            if triplet in matched:
                group_valid = False
                break
        
        # If no conflicts, mark all triplets in this group as matched
        if group_valid:
            for triplet in group:
                matched.add(triplet)
            total_triplets_matched += len(group)
    
    # Return the fraction of triplets matched out of total triplets
    return total_triplets_matched / (n_agents if n_agents > 0 else 1)

def score_toggles(solution, groupings):
    # Create a matched array with the size of triplets (n_agents), not groupings
    matched = set()
    total_triplets_matched = 0
    
    for used, group in zip(solution, groupings):
        if used:
            # Check if any triplet in this group is already matched
            group_valid = True
            for triplet in group:
                if triplet in matched:
                    group_valid = False
                    break
            
            # If no conflicts, mark all triplets in this group as matched
            if group_valid:
                for triplet in group:
                    matched.add(triplet)
                total_triplets_matched += len(group)
    
    # Return the fraction of triplets matched
    return total_triplets_matched / (n_agents if n_agents > 0 else 1)

def generate_solutions_toggles(solutions, groupings, scores, elitism_keep=5, num_changes=5, pop_size=50):
    # Keep best solutions from previous generation
    elite_indices = np.argsort(scores)[-elitism_keep:]
    new_solutions = [solutions[i] for i in elite_indices]
    new_scores = [scores[i] for i in elite_indices]
    
    # Generate new solutions
    while len(new_solutions) < pop_size:
        parent_idx = np.random.choice(len(solutions), p=np.maximum(scores, 0)/max(sum(np.maximum(scores, 0)), 1e-10))
        child = toggle_elements(solutions[parent_idx], num_changes)
        child_score = score_toggles(child, groupings)
        new_solutions.append(child)
        new_scores.append(child_score)
    
    return new_solutions, new_scores

def generate_solution_permutations(solutions, groupings, scores, elitism_keep=5, num_changes=5, pop_size=50):
    # Keep best solutions from previous generation
    elite_indices = np.argsort(scores)[-elitism_keep:]
    new_solutions = [solutions[i] for i in elite_indices]
    new_scores = [scores[i] for i in elite_indices]
    
    # Generate new solutions
    while len(new_solutions) < pop_size:
        parent_idx = np.random.choice(len(solutions), p=np.maximum(scores, 0)/max(sum(np.maximum(scores, 0)), 1e-10))
        child = permute_grouping(solutions[parent_idx], num_changes)
        child_score = score_permutation(child, groupings)
        new_solutions.append(child)
        new_scores.append(child_score)
    
    return new_solutions, new_scores


if __name__ == "__main__":
    # Create environments
    num_environments = 16  # Changed from 5 to 16 as requested
    generations = 50
    environments = [create_environment(n_agents) for _ in range(num_environments)]
    toggle_results = []
    permutation_results = []
    
    for env_idx, triplets in enumerate(tqdm(environments)):
        # Get valid groupings
        groupings = find_valid_groupings(triplets)
        print("Number of valid groupings: ", len(groupings))
        
        # Optimize with toggles
        toggle_solutions = [np.random.randint(0, 2, size=len(groupings)) for _ in range(50)]
        toggle_scores = [score_toggles(sol, groupings) for sol in toggle_solutions]
        for _ in tqdm(range(generations), desc="Toggle generations"):
            toggle_solutions, toggle_scores = generate_solutions_toggles(toggle_solutions, groupings, toggle_scores)
        toggle_results.append(max(toggle_scores))
        
        # Optimize with permutations
        perm_solutions = [np.random.permutation(len(groupings)) for _ in range(50)]
        perm_scores = [score_permutation(sol, groupings) for sol in perm_solutions]
        for _ in tqdm(range(generations), desc="Permutation generations"):
            perm_solutions, perm_scores = generate_solution_permutations(perm_solutions, groupings, perm_scores)
        permutation_results.append(max(perm_scores))
    
    # Aggregate results
    print(f"Toggle method average: {np.mean(toggle_results):.4f}")
    print(f"Permutation method average: {np.mean(permutation_results):.4f}")
    print(f"Better method: {'Toggle' if np.mean(toggle_results) > np.mean(permutation_results) else 'Permutation'}")
    
    # Graph the results
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_environments)
    width = 0.35
    
    # Create bars
    toggle_bars = ax.bar(x - width/2, toggle_results, width, label='Toggle Method')
    perm_bars = ax.bar(x + width/2, permutation_results, width, label='Permutation Method')
    
    # Add horizontal lines for averages
    ax.axhline(y=np.mean(toggle_results), color='blue', linestyle='--', alpha=0.7, label='Toggle Average')
    ax.axhline(y=np.mean(permutation_results), color='orange', linestyle='--', alpha=0.7, label='Permutation Average')
    
    # Add labels and title
    ax.set_xlabel('Environment Index')
    ax.set_ylabel('Performance (fraction of triplets matched)')
    ax.set_title('Comparison of Optimization Methods Across Environments')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_environments)])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/dual_donor_optimization_comparison.png')
    plt.show()