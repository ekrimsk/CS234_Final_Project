     

import numpy as np
import os, sys, time
import pickle
import glob 
import shutil 
import matplotlib.pyplot as plt 

from util import data_process
from data_process import Data_handler
from util.data import load 
# Import some things from utils for plotting


# NOTE -- we will probably want to move almost all of this out of here!!!




def main():
    #file = 'dir_list.pkl'       # may want a parent dir or something 





    # First thing we want to put in here is just being able to unpicklle the file and get the directory list 
    #with open(file, "rb") as f:
    #    dir_list = pickle.load(f)

    # manaully writing dir list 
    # dir_list = ["Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/results/ERA5-v0_k_3_episodes_2000"]
            
    # loop through all the directories and do some tpye of processing 
    #https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

    #color_list = ["r", "b", "g", "c", "m", "y", "k", "w"]
    #  Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/

    # maybe add one with 3 just because pretty easy??? 

    dir_list_3 = ["results/ERA3-v0_k_1_episodes_2000",
                  "results/ERA3-v0_k_4_episodes_2000",
                  "results/ERA3-v0_k_8_episodes_2000"]


    dir_list_5 = ["results/ERA5-v0_k_1_episodes_2000",
                  "results/ERA5-v0_k_3_episodes_2000", 
                  "results/ERA5-v0_k_16_episodes_2000",
                  "results/ERA5-v0_k_32_episodes_2000"]

    #dir_list_8 = ["results/ERA8-v0_k_1_episodes_4000",
    #              "results/ERA8-v0_k_25_episodes_4000", 
    #              "results/ERA8-v0_k_128_episodes_4000", 
    #              "results/ERA8-v0_k_256_episodes_4000"]     


    #dir_list_8 = ["results/ERA8-v0_k_1_episodes_6000",
    #              "results/ERA8-v0_k_25_episodes_6000", 
    #              "results/ERA8-v0_k_128_episodes_6000", 
    #              "results/ERA8-v0_k_256_episodes_6000"]  

    dir_list_8 = ["results/ERA8-v0_k_1_episodes_8000",
                  "results/ERA8-v0_k_25_episodes_8000", 
                  "results/ERA8-v0_k_128_episodes_8000", 
                  "results/ERA8-v0_k_256_episodes_8000"]                                        



    dir_list_10 = ["results/ERA10-v0_k_1_episodes_16000",
                   "results/ERA10-v0_k_102_episodes_16000", 
                    "results/ERA10-v0_k_512_episodes_16000", 
                   "results/ERA10-v0_k_1024_episodes_16000"]        

    end3, max3 = plot_dir_list(dir_list_3, 'plots/plots_3')
    end5, max5 = plot_dir_list(dir_list_5, 'plots/plots_5')
    #plot_dir_list(dir_list_8, 'plots/plots_8_4000')
    #plot_dir_list(dir_list_8, 'plots/plots_8_6000')
    end8, max8 = plot_dir_list(dir_list_8, 'plots/plots_8_8000')
    end10, max10 = plot_dir_list(dir_list_10, 'plots/plots_10_16000')

    print("========== n = 3 Data ===================")
    print("END: ", end3)
    print("MAX: ", max3)
    print("========== n = 5 Data ===================")
    print("END: ", end5)
    print("MAX: ", max5)
    print("========== n = 8 Data ===================")
    print("END: ", end8)
    print("MAX: ", max8)
    print("========== n = 10 Data ===================")
    print("END: ", end10)
    print("MAX: ", max10)


    # create a directory for outputing plots to 


def plot_dir_list(dir_list, save_name):

    # will want to be able to plot groups!!! -- will have n = 5, n =8 groups with different k values 
    # parse out this info from the file path -- could make lists of data handlers 


    # create consistent ordered color lists 
    avg_list = []
    gen_list = []
    std_list = []
    label_list = []
    max_list = []
    end_list = []

    # shade_color_list = # can this just be a function of original plot color?? 


    for path in dir_list:
        print('Processing: ', path)
        zips = glob.glob(path + '/*.zip') 
        zipf = zips[0]
        # print(zipf)

        path_parts = os.path.split(path)
        path_end = path_parts[-1]
        splits = path_end.split("_")
        env_name = splits[0]
        knn = int(splits[2])
        eps = int(splits[4])


        print("Path: ", path, " env: ", env_name, "knn: ", knn, " eps: ", eps)
        #label = "k = {}, ({0:05.3f}%)".format(knn, knn/)
        label = "k = {}".format(knn)

        print(label)
        # parse it out  


        #data = load(zipf)
        dh = Data_handler(zipf)
        # dh.plot_rewards()           # so overwrite this stuff because we dont need all the gtk business 


        # get the rrwards 
        rewards = dh.get_full_episode_rewards() 
        batch_size = 64 # I THINK?????? -- doesnt really matter 

        avg = np.array(data_process.apply_func_to_window(rewards, batch_size, np.average))
        max_val = max(avg)
        end_val = avg[-1]
        std = np.array(data_process.apply_func_to_window(rewards, batch_size, np.std))
        gen = np.arange(0, len(avg))


        max_list.append(max_val)
        end_list.append(end_val)
        avg_list.append(avg)
        gen_list.append(gen)
        std_list.append(std/2)
        label_list.append(label)

        # append to all the lists 

    ml_plot(gen_list, avg_list, std_list, label_list, save_name=save_name)  # need to handle color shading too 
    return end_list, max_list




def ml_plot(gen_list, mean_list, std_list, label_list, save_name=None, color_list=None): # ADD MORE 

    # check if lists -- if so, loop thorugh (or maybe just do this in the caller )


    if color_list is None:  # could implement option to overwrite it 
        # https://matplotlib.org/gallery/color/named_colors.html
        color_list = ['darkred', 'darkblue', 'darkgreen', 'peru'] # 'darkred', 
  


    fig = plt.figure()  # new figure for no 


    # find the maximum in the gen list to set the xlimit with 

    # y limit -- -300 to 1000 ? 


    for gen, mean, std, label, col in zip(gen_list, mean_list, std_list, label_list, color_list):    
        plt.plot(gen, mean, color=col,  label=label)
        plt.fill_between(gen, mean - std, mean + std, color=col, alpha=0.2)
            

    # TODO -- this stuff 
    # plt.title("foobar") # dont want titles though??? 
    plt.xlabel("episodes")
    plt.ylabel("score")
    plt.legend(loc='best')      # probably bottom right 
    plt.ylim(-300, 150)

    #plt.show()

    # TODO -- set some axis limits 
    plt.savefig(save_name + '.png', bbox_inches='tight')





if __name__ == '__main__':
    main()

