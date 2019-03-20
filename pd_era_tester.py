import gym
import numpy as np
import os, sys
import time


import utils 
from config import get_config


env_name = 'ERA5-v0'



def main():


    p_list = [1.75, 2, 2.25, 2.5, 2.75, 3]
    d_list = [0, 0.05, 0.1, 0.3, 0.7, 0.8, 0.9, 1, 1.1, 1.2]


    env = gym.make(env_name)
    max_ep_len = env.spec.timestep_limit
    num_episodes = 200

    best_val = -np.Inf
    for p in p_list:
        for d in d_list:
            episode_reward_list = []



            total_rewards = 0

            for episode in range(num_episodes):

                obs = env.reset()
                episode_rew = 0
                for t in range(max_ep_len):
                    action = env.unwrapped.act_pd()
                    obs, rew, done, info = env.step(action)
                    episode_rew += rew
                     
                    if done:
                        break
                        #print(t, "done reason: ", info["done_reaon"])
                        #print(info)
                        #break

                #print('Episode reward', episode_rew)
                #print('Final Position: ', obs[2 * env.n], '\n\n')
                episode_reward_list.append(episode_rew)
                total_rewards += episode_rew

            mean_reward = total_rewards/num_episodes
            if (mean_reward > best_val):
                best_val = mean_reward
                print('Mean episode reward: {}'.format(mean_reward))
                print('p: ', p, ' d: ', d)
                # print(episode_reward_list)


if __name__ == '__main__':
    main()