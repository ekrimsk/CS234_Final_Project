import gym
import numpy as np
import os, sys
import time


import utils 
from config import get_config

# TODO ===== MOVE ALL THIS OUT OF HERE AND DELETE THIS FILE 
"""
path_agent_parent_dir = '../'  
sys.path.append(path_agent_parent_dir + '../')
sys.path.append(os.path.dirname('bdq') + path_agent_parent_dir)
path_logs = path_agent_parent_dir + 'bdq/' 

import envs
from bdq import deepq
"""

env_name = 'ERA5-v0'


# Load in the configs and establish a baseline for the greedy algorithm

# TODO --- add this as a class method??????




def main():
    baseline = True # why not 
    config = get_config('era', baseline)


    env = gym.make(env_name)
    env = env.unwrapped

    env.unwrapped.torque_plot()

    total_rewards = 0
    num_episodes = 20
    #num_episodes = 1000
    #max_ep_len = config.max_ep_len
    max_ep_len = env.spec.timestep_limit 

    init_env = env
    record_path = "./test_vid_greedy"

    for episode in range(num_episodes):

        uid = "greeedy_test_{0:05d}".format(episode)
        env = gym.wrappers.Monitor(init_env, record_path, video_callable=lambda x: True, resume=True, uid=uid)

        #obs, done = env.reset(verbose=True), False
        #obs = env.unwrapped.reset(verbose=True)
        obs = env.reset()
        done = False
        #obs = env.reset()

        #print("init obs: ", obs)
        done = False
        episode_rew = 0
        for t in range(max_ep_len):
            env.render()
            time.sleep(.05)
            greedy_action = env.unwrapped.act_greedy()

            obs, rew, done, info = env.step(greedy_action)
            #print('greedy_action: ', greedy_action)
            episode_rew += rew
            #print(f)
            if done:
                #print(t, "done reason: ", info["done_reaon"])
                #print(info)
                #print("obs: ", obs)
                break


        #print('Episode reward', episode_rew)
        #print('Final Position: ', obs[2 * env.n], '\n\n')

        total_rewards += episode_rew

    print('Mean episode reward: {}'.format(total_rewards/num_episodes))

if __name__ == '__main__':
    main()