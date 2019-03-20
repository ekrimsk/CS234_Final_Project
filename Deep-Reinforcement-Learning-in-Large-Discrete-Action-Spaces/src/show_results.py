#!/usr/bin/python3
import numpy as np
from util.data_process import *


# EDITED 


def show(folder='',
        episodes=2500,
        actions=32,
        k=3,
        experiment='ERA5-v0'):
    v = 3   # no clue what this is 
    id = 0  # keep? 

    name = 'results/obj/{}data_{}_Wolp{}_{}{}k{}#{}.json.zip'.format(folder,
                                                                     episodes,
                                                                     v,
                                                                     experiment[:3],
                                                                     actions,
                                                                     k,
                                                                     id
                                                                     )

    data_process = Data_handler(name)

    print("Data file is loaded")

    data_process.plot_rewards()
    data_process.plot_average_reward()
    data_process.plot_action_distribution()
    data_process.plot_action_distribution_over_time()
    data_process.plot_action_error()


if __name__ == '__main__':
    show()
