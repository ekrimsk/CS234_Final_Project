# Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
Link to [paper](https://arxiv.org/abs/1512.07679)

Implementation of the algorithm in Python 3, TensorFlow and OpenAI Gym.



This paper introduces Wolpertinger training algorithm that extends the Deep Deterministic Policy Gradient training algorithm introduced in [this](https://arxiv.org/abs/1509.02971) paper.

I used and extended  **stevenpjg**'s implementation of **DDPG** algorithm found [here](https://github.com/stevenpjg/ddpg-aigym) licensed under the MIT license.

Master is currently **only for continuous action spaces**.

The branch discrete-and-continuous provides the ability to use the discrete environments of the gym. 


# Changes by ekrimsk
The README above is for the discrete and continunous branch of the repo at (https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/tree/discrete-and-continuous)




The following files were added:
    - generate_plots.py
    - test_model_load.py

Additional functionality has been added to the agent superclass in DDPG to allow for saving and restoring trained agents 

Additionally, the dependencies on FLANN have been removed from action_space as they are not needed when all actions are 1D integers (and were causing segfaults)

