
# well have a bunch of imports here 


import gym
import numpy as np
import os, sys
import time
import matplotlib.pyplot as plt 


import utils # to register environments 

sys.path.insert(0, './Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/src')


from wolp_agent import *
from ddpg.agent import DDPGAgent

def test_trained_agents():      # no inputs for now but may want to change that and add them in main 


    trained_agents = []
    environment_list = []

    dd = './Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/'


    dir_list_3 = [dd + "results/ERA3-v0_k_1_episodes_2000",
                  dd + "results/ERA3-v0_k_4_episodes_2000",
                  dd + "results/ERA3-v0_k_8_episodes_2000"]


    dir_list_5 = [dd + "results/ERA5-v0_k_1_episodes_2000",
                  dd + "results/ERA5-v0_k_3_episodes_2000", 
                  dd + "results/ERA5-v0_k_16_episodes_2000",
                  dd + "results/ERA5-v0_k_32_episodes_2000"]

    #dir_list_8 = ["results/ERA8-v0_k_1_episodes_4000",
    #              "results/ERA8-v0_k_25_episodes_4000", 
    #              "results/ERA8-v0_k_128_episodes_4000", 
    #              "results/ERA8-v0_k_256_episodes_4000"]     


    #dir_list_8 = ["results/ERA8-v0_k_1_episodes_6000",
    #              "results/ERA8-v0_k_25_episodes_6000", 
    #              "results/ERA8-v0_k_128_episodes_6000", 
    #              "results/ERA8-v0_k_256_episodes_6000"]  

    dir_list_8 = [dd + "results/ERA8-v0_k_1_episodes_8000",
                  dd + "results/ERA8-v0_k_25_episodes_8000", 
                  dd + "results/ERA8-v0_k_128_episodes_8000", 
                  dd + "results/ERA8-v0_k_256_episodes_8000"]                                        



    dir_list_10 = [dd + "results/ERA10-v0_k_1_episodes_16000",
                   dd + "results/ERA10-v0_k_102_episodes_16000", 
                   dd +  "results/ERA10-v0_k_512_episodes_16000", 
                   dd + "results/ERA10-v0_k_1024_episodes_16000"]     

    n_list = [3, 5, 8, 10]

    # Create list of initial observations that we will use to test each agent 
    env = gym.make('ERA5-v0')   # Doesn't matter which we use 
    num_test_eps = 100   # episodes to check behavior with     
    init_obs_list = []
    for _ in range(num_test_eps):
        obs = env.reset()   # this will be with however "env" was defined -- 
        init_obs_list.append(obs)              

    # for each dir list -- restore trained agents and dump into new list 


    dir_lists = [dir_list_3, dir_list_5, dir_list_8, dir_list_10]


    trained_agent_lists = []
    environment_lists = []


    # TODO -- make this a function call
    for dir_list in dir_lists:
        trained_agent_list = []
        environment_list = []
        for path in dir_list:       # TODO == actually make the list of paths 

            # may want to move some if this parsing to somewhere where it can be reused -- would be good for the gen plot code 
            path_parts = os.path.split(path)
            path_end = path_parts[-1]
            splits = path_end.split("_")
            env_name = splits[0]
            knn = int(splits[2])
            eps = int(splits[4])

            print("Path: ", path, " env: ", env_name, "knn: ", knn, " eps: ", eps)

            # will need to check if this is the right number -- will probably need to parse the folder somehow 
            env = gym.make(env_name)
            restore_agent = WolpertingerAgent(env) # other args are optional 

            print("Trying to restore agent")
            
            restore_agent.restore(path); #print(restore_agent.action_space._dimensions) #restore_agent.action_space.rebuild_flann()
            print("Agent restored!"); #  [print(v) for v in vars(restore_agent.action_space).items()]

            environment_list.append(env)
            trained_agent_list.append(restore_agent)

        trained_agent_lists.append(trained_agent_list)
        environment_lists.append(environment_list)    


   
    # create a list of dicts!! == each dict has actions and observations 

    total_trained_results = [] 
    total_greedy_results = []
    total_pd_results = []

    # loops through different n numbers 
    for trained_agent_list, environment_list, n in zip(trained_agent_lists, environment_lists, n_list):

        trained_results = []    # list of results 
        greedy_results = []
        pd_results = []

        print('---------------- ', str(n), '---------------------------')
        for agent, env in zip(trained_agent_list, environment_list):        
            trained_agent_results = []    # list of results 
            greedy_agent_results = []
            pd_agent_results = [] 

            # Run all the sims on trained agents + greedy agents 
            trained_rewards = 0
            greedy_rewards = 0
            pd_rewards = 0
            for init_obs in init_obs_list:   # trying all the different initial conditions 
                # call method -- run episode 
                
                actions, observations, rewards, info = run_episode(env, agent, init_obs)

                #print("ACTIONS: ", actions)
                #print("OBSERVATIONS: ", observations)
                #print("REWARDS: ", rewards)
                #print(info)

                total_trained_reward = sum(rewards)
                trained_rewards += total_trained_reward
                res_dict = {"actions" : actions,
                            "observations" : observations,
                            "rewards" : rewards}

                trained_agent_results.append(res_dict) # want per agent though 

                # reset the environment correctly 
                actions, observations, rewards, info = run_episode(env, agent, init_obs, greedy=True)

                total_greedy_reward = sum(rewards)
                greedy_rewards += total_greedy_reward
                res_dict = {"actions" : actions,
                            "observations" : observations,
                            "rewards" : rewards}

                greedy_agent_results.append(res_dict) # want per agent though 


                actions, observations, rewards, info = run_episode(env, agent, init_obs, pd=True)
                total_pd_reward = sum(rewards)
                pd_rewards += total_pd_reward
                res_dict = {"actions" : actions,
                            "observations" : observations,
                            "rewards" : rewards}

                pd_agent_results.append(res_dict) # want per agent though 

                 # print("Trained agent rewards: ", total_trained_reward, " Greedy agent rewards: ",  total_greedy_reward)


            mean_trained = trained_rewards/len(init_obs_list) 
            mean_greedy = greedy_rewards/len(init_obs_list)
            mean_pd = pd_rewards/len(init_obs_list)

            print("n: ", n)
            print("Trained: ", mean_trained)
            print("Greedy: ", mean_greedy)
            print("PD: ", mean_pd)



            trained_results.append(trained_agent_results)    
            greedy_results.append(greedy_agent_results)



        total_trained_results.append(trained_results)
        total_greedy_results.append(greedy_results)



    """

    # Now we have all the data -- can iterate over and characterize 
    num_plot = 5
    theta_thresh = 0.5 
    for trained_agent_results, greedy_agent_results in zip(trained_results, greedy_results):
        

        # just plot the trajectory for the first 
        #for i in range(len(trained_agent_results)):
        i = 0 
        while (i < num_plot) 
            trained_dict = trained_agent_results[i]
            greedy_dict = greedy_agent_results[i]

            # will want to get the thetas for each of these from the observations -- make function for it we call but do it here quick and dirty now 
            theta, theta_goal = get_thetas(trained_dict)


            # 
            if (abs(theta[0] - theta_goal) > theta_thresh): 
                i += 1 
                plot_trajectories(theta, theta_goal, i)


    """

    # could make a table -- think about what we want to actually put in it 

    # lets plot the first one that looks succesful -- (well dont have rewards recorded here)



    # also make some plots for trajectories -- which sims to look at?  -- for now, just look at first sim -- move this out to seperate function 
    # for the same n number -- would be good to see how different k numbers compare 

    # will want to do for a handful of them 
    #for (trained_agent_results, greedy_agent_results) in zip(results, greedy_results):


    # make inidividual trajectory plot for first in each 




    # maybe llook at distribution of acitons too  

def get_thetas(episode_dict):
    observations = episode_dict["observations"]

    theta = []
    for obs in observations:
        theta.append(obs[-3])
        theta_goal = obs[-1]

    return theta, theta_goal


# Will want some trajectory plotting code as well too! -- show rangees! 

# May even wanto move this to an era method so it can access all the ERA params 

def plot_trajectories(theta, theta_goal, name=None, colors=None):   # may want to make this accept lists of trajectories to plot multiple 

    # Could check if we were passed in lists of trajectories and then iterate over them 


    # TODO =  as a check in the caller, dont plot boring trajectories -- make sure goal and init theta are sufficiently far away

    # TODO -- maybe change to degrees?? (easier to understand)
    theta = np.array(theta)

    dt = 1/20 
    t = dt * np.arange(0, len(theta))    # TODO -- scale it properly 

    # theta_goal 

    fig = plt.figure()  # new figure for no 

    plt.plot(t, theta)  # black, thick, solid (unless we do multiple in which case use differnt cololros )

    theta_max = np.ones_like(t) * np.deg2rad(45)
    theta_min = np.ones_like(t) * np.deg2rad(-85)       

    plt.plot(t, theta_max, 'k--') 
    plt.plot(t, theta_min, 'k--') 


    #plt.plot(t, theta, color=color,  label=name)

    # Add Theta bounds to the plot

    # could potentially plot like 3 trajectories together if sufficiently different -- just use differnt colors 


    # TODO -add dashed line for plotting theta goal 
    # TODO - maybe add cone showing acceptable final velocity         

    # plt.title("foobar") # dont want titles though??? 
    plt.xlabel("time")
    plt.ylabel(r'$\theta$')
    # plt.legend(loc='best')

    plt.xlim(0, max(t))
    #plt.show()


    if name is None:
        plt.savefig('foo.png', bbox_inches='tight')
    else:
        plt.savefig(str(name) + '.png', bbox_inches='tight')        # make naming not dumb though 






def run_episode(env, agent, init_obs, greedy=False, pd=False):       # some way of passing in the original observation to reset it 
                                            # do we also want to deal with passing in maximum episode length>>> 

    #if greedy:
    #    print('================ RUNNING GREEDY EPISODE =================================')
    #else:
    #    print('================ RUNNING TRAINED EPISODE =================================')
            
        


    # -- really just last couple parts of observatoin array 
    env = env.unwrapped
    tmp = env.reset()     # reset the environment 
    #n = int((len(tmp) - 3)/2) # why cant access n???????? 
    n = env.get_n() 
    r = env.get_r()

    goal = init_obs[-1]
    theta_dot_0 = init_obs[-2]
    theta_0 = init_obs[-3]

    # set the goal 
    env.set_goal(goal)

    # set the init state
    init_state = np.zeros((2*n + 2,))
    init_state[0:n] = 0 # clutches off 
    init_state[n:(2*n)] = -theta_0 * r
    init_state[2*n] = theta_0
    init_state[(2*n) + 1] = theta_dot_0
    env.set_state(init_state)

    actions = []    # just do as a list 
    observations = []    # as a list 
    rewards = []    # as a list 


    max_steps = env.spec.timestep_limit    # -- may need to move this code up to before unwrapping 
    obs = np.concatenate([init_state, [goal]])  

  
    #print("max steps: ", max_steps)
    for _ in range(max_steps):
        if greedy:
            action = env.act_greedy()
        elif pd:
            action = env.act_pd()
        else:
            action = np.asscalar(agent.act(obs))

        obs, rew, done, info = env.step(action)

        #print(action)
        actions.append(action)
        observations.append(obs)
        rewards.append(rew)
        # could add rewards if we want too 


        if done:
            break 

    #print('================ EPISODE DONE =================================')

    return actions, observations, rewards, info


if __name__ == '__main__':
    test_trained_agents()

