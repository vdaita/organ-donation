from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN, SAC
import matplotlib.pyplot as plt
import numpy as np
from binary_decision_environment import BinaryDecisionEnvironment
  
def make_env(rank):
    def _init():
        env = BinaryDecisionEnvironment(n_agents=100, n_timesteps=32)
        env.reset(seed=rank)
        return env
    return _init

if __name__  == "__main__":
    # model = RecurrentPPO("MlpLstmPolicy", DummyVecEnv([lambda: BinaryDecisionEnvironment(n_agents=250)]), verbose=1)
    n_envs = 16
    # env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = DummyVecEnv([lambda: BinaryDecisionEnvironment(n_agents=100, n_timesteps=32)])
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.995, tensorboard_log="./tb_runs/")
    model.learn(total_timesteps=300000)

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

        model_rewards.append(np.sum(env.matched_agents) / env.n_agents)
        greedy_reward = env.get_greedy_result()
        greedy_rewards.append(greedy_reward)

    ratio = [model_reward / greedy_reward for model_reward, greedy_reward in zip(model_rewards, greedy_rewards)]
    print(f"Model rewards: {model_rewards}")
    print(f"Greedy rewards: {greedy_rewards}")
    print(f"Ratio: {ratio}")

    plt.boxplot(ratio)
    plt.show()
   