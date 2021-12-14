import gym
from stable_baselines3 import PPO

import PredPrey
from sb_predprey.PredPrey.ActorCriticPolicyMod import ActorCriticPolicyMod

if __name__ == "__main__":
    env = gym.make('PredPreyEnv-v0')
    observation = env.reset()
    model = PPO(ActorCriticPolicyMod, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("test")

    for _ in range(1000):
        # Action vector is list of 3 action for each robot. Total length 6
        observation, reward, done, info = env.step([1, 2, 3])
        print(observation)
        env.render()
        if (done):
            break
    env.close()
