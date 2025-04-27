from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import numpy as np
from binary_decision_environment import BinaryDecisionEnvironment
  
if __name__  == "__main__":
    model = RecurrentPPO("MlpLstmPolicy", DummyVecEnv([lambda: BinaryDecisionEnvironment(n_agents=250)]), verbose=1)
    model.learn(total_timesteps=2000)

    num_runs = 16
    model_rewards = []
    greedy_rewards = []
    for _ in range(num_runs):
        env = BinaryDecisionEnvironment()
        seed = np.random.randint(0, 2 ** 32 - 1)

        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action > 0.5)
        model_rewards.append(reward)
        greedy_reward = env.get_greedy_result()
        greedy_rewards.append(greedy_reward)

    ratio = [model_reward / greedy_reward for model_reward, greedy_reward in zip(model_rewards, greedy_rewards)]
    print(f"Model rewards: {model_rewards}")
    print(f"Greedy rewards: {greedy_rewards}")
    print(f"Ratio: {ratio}")

    plt.boxplot(ratio)
    plt.show()
   