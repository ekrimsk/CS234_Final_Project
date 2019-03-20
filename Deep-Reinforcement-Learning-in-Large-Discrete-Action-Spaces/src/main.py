#!/usr/bin/python3

# NOTE -- pulled from branch for discrete 

import time # for render -- should delete
import os, sys
import pickle
# https://stackoverflow.com/questions/16780014/import-file-from-parent-directory
# importing from tweo dirtectories up 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils # IN the parent directory
import gym

import numpy as np

from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data

from util.timer import Timer


# CHECK GPU USAGE!!!!!!!!!!!!!!!!!!!!!!!!!!


def run(episodes=2500,   #2500,
        render=False,
        experiment='ERA5-v0',            
        max_actions=1000,
        knn=0.1,
        save_dir=None):

    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit     # pulls from the init file where it is registered 

    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn)

    timer = Timer()

    data = util.data.Data()
    data.set_agent(agent.get_name(), int(agent.action_space.get_number_of_actions()),
                   agent.k_nearest_neighbors, 3)
    data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)

    agent.add_data_fetch(data)
    print(data.get_file_name())

    full_epoch_timer = Timer()
    reward_sum = 0


    # EREZ ADD 
    num_avg = 40
    recent_rew_list = []    

    for ep in range(episodes):

        timer.reset()
        observation = env.reset()

        total_reward = 0
        print('Episode ', ep, '/', episodes - 1, 'started...', end='')



        for t in range(steps):

            if render:
                env.render()

            action = agent.act(observation)

            data.set_action(action.tolist())

            data.set_state(observation.tolist())

            prev_observation = observation
            observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)

            data.set_reward(reward)

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            agent.observe(episode)

            total_reward += reward

            if done or (t == steps - 1):
                t += 1
                reward_sum += total_reward
                time_passed = timer.get_time()

                # NOTE: shouldnt we be reporting average over recent? 
                # Added better print formating 
                #print('\tReward:{:04.4f} \tSteps:{} \tt:{} \t({}/step) \tCur avg={:04.4f}'.format(total_reward, t,
                #                                                            time_passed, round(
                #                                                                time_passed / t),
                #                                                            reward_sum / (ep + 1)))

                # EREZ ADDED 
                recent_rew_list.append(total_reward)

                if ep < num_avg:
                    recent_avg = sum(recent_rew_list)/len(recent_rew_list)
                else:
                    recent_avg = sum(recent_rew_list[-num_avg:])/num_avg

                print('\tReward:{:05.3f} \tSteps:{} \tt:{} \t({}/step) \tCur avg={:04.4f}'.format(total_reward, t,
                                                                            time_passed, round(
                                                                                time_passed / t),
                                                                                recent_avg))
                #print('\tReward:{:04.4f} \tSteps:{} \tt:{} \t({}/step) \tCur avg={:04.4f}'.format(total_reward, t,
                #                                                            time_passed, round(
                #                                                                time_passed / t),
                #                                                            round(reward_sum / (ep + 1))))

                # TODO -- look into these and change how we write out directories 
                data.finish_and_store_episode()

                break
    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    #data.save()

    if save_dir is None:
        data.save() # EREZ ADDED ARG 
    else:
        data.save(path=save_dir) # EREZ ADDED ARG 
        agent.save(save_dir)  # could add a seperate call to data,save from within this!


    # Code added below 
    return agent 

if __name__ == '__main__':

    # list of k nearest to try 
    #k_try = [1e-6, 0.05, 0.1]   # 0.1 might be agressive if we go up to 2^16
    #k_try = [1e-6, .01, 0.1]   # just for debug purposed 
    k_try = [1e-6, 0.1, 0.5, 1]     # 1e-6 corresponds to k = 1
    #k_try = [1]
    #n_list = [5, 8, 10, 12, 16]
    #n_list = [8, 10, 16]   
    #n_list = [8 ]
    # n_list = [10]

    n_list = [3]
    num_episodes = [2000]
    # 2000 good for n = 5, k = 3 -- may need another 1000 for n = 8 
    #num_episodes = [int(1e4), int(1e4), int(1e4), int(1e4), int(1e4)]
    #num_episodes = [5000, 5000, 5000]
    #num_episodes = [2000, 3000]
    #num_episodes = [8000]
    #num_episodes = [16000]
    #num_episodes = [20000]

    dir_list = []   # list of directories we are going to save into 

    for k_ratio in k_try:
        for n, episodes in zip(n_list, num_episodes):
            num_actions = 2**n
            env_name = 'ERA' + str(n) + "-v0"

            knns = k_ratio * num_actions    # to use as a check if running more than 10 with high neighbors 
            # some checks to break if the resulting numbers are absurd 


            knn = max(1, int(num_actions * k_ratio)) 
            path = "./results/{}_k_{}_episodes_{}".format(env_name, knn, episodes)
            dir_list.append(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)  # ADDED

            print("Saving Results to: ", path)

            run(episodes=episodes,
                render=False,
                experiment=env_name,            
                max_actions=1e6,
                knn=k_ratio,
                save_dir=path)

    # Need a good processing pipeline 

    file = 'dir_list.pkl'   
    with open(file, "wb") as f:
            pickle.dump(dir_list, f, pickle.HIGHEST_PROTOCOL)
