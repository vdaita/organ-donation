from matplotlib import pyplot as plt
import numpy
import json

with open("model_results.json") as f:
    data = f.read()
    data = json.loads(data)

    # patient ratio  
    patient_ratio = [(x / y) for x, y in zip(data["reward"], data["patient_reward"])]

    # greedy ratio
    greedy_ratio = [(x / y) for x, y in zip(data["reward"], data["greedy_reward"])]

    # Plot the reward ratios
    plt.figure(figsize=(10, 6))
    plt.boxplot([patient_ratio, greedy_ratio], labels=["Model Reward / Patient Reward", "Model Reward / Greedy Reward"])
    plt.ylabel("Reward Ratio")
    plt.title("Reward Ratios Distribution")
    plt.grid(True)
    plt.savefig("model_reward_ratios_boxplot.png")
    plt.show()