import json
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire

SHOW_OPTIMAL = True

def make_graph(SHOW_OPTIMAL=True):
    results = json.load(open("simulation_results.json", "r"))
    results_by_agent_count = {}

    for result in results:
        if result["num_agents"] not in results_by_agent_count:
            results_by_agent_count[result["num_agents"]] = []
        results_by_agent_count[result["num_agents"]].append(result)


    for agent_count in results_by_agent_count:
        aggregated_results = {}

        for result in results_by_agent_count[agent_count]:
            greedy_reward = [res for res in result["results"] if res["type"]["method"] == "greedy"][0]["reward"]
            periodic_rewards = [res for res in result["results"] if res["type"]["method"] == "periodic"]
            mixed_rewards = [res for res in result["results"] if res["type"]["method"] == "mixed"]

            for periodic_reward in periodic_rewards:
                name = periodic_reward["type"]["method"] + "-" + str(periodic_reward["type"]["period"])
                ratio = periodic_reward["reward"] / greedy_reward
                if name not in aggregated_results:
                    aggregated_results[name] = []
                aggregated_results[name].append(ratio)
            
            for mixed_reward in mixed_rewards:
                name = mixed_reward["type"]["method"] + "-" + str(mixed_reward["type"]["period"])
                ratio = mixed_reward["reward"] / greedy_reward
                if name not in aggregated_results:
                    aggregated_results[name] = []
                aggregated_results[name].append(ratio)
            
            if SHOW_OPTIMAL:
                if "retrospective-optimal" not in aggregated_results:
                    aggregated_results["retrospective-optimal"] = []
                aggregated_results["retrospective-optimal"].append(1 / greedy_reward)

        # box plot (w/ claude)
        plt.figure(figsize=(12, 6))
        labels = list(aggregated_results.keys())
        data = [aggregated_results[key] for key in labels]
        
        plt.boxplot(data, labels=labels)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Greedy baseline')
        plt.title(f'Reward Ratios vs. Greedy Method (Agent Count: {agent_count})')
        plt.ylabel('Ratio of Reward to Greedy (log scale)')
        plt.xlabel('Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()

        optimal_str = "_with_optimal" if SHOW_OPTIMAL else ""

        plt.savefig(f'reward_ratios_{agent_count}_agents{optimal_str}.png')
        plt.show()