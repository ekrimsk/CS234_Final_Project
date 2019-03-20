from gym.envs.registration import register

# MuJoCO environments
#register(id='Reacher3DOF-v0', entry_point='envs.mujoco.reacher_3dof:Reacher3DOFEnv', max_episode_steps=50)
#register(id='Reacher4DOF-v0', entry_point='envs.mujoco.reacher_4dof:Reacher4DOFEnv', max_episode_steps=60)
#register(id='Reacher5DOF-v0', entry_point='envs.mujoco.reacher_5dof:Reacher5DOFEnv', max_episode_steps=70)
#register(id='Reacher6DOF-v0', entry_point='envs.mujoco.reacher_6dof:Reacher6DOFEnv', max_episode_steps=80)


# Erez Added 
# envs.something is filepath
register(id='ERA-v0', entry_point='era:ERAEnv', max_episode_steps=100)  # will need to change 

# trying these as sublcassses of an ERA super class
register(id='ERA1-v0', entry_point='era:ERA1Env', max_episode_steps=100)  # will need to change 
register(id='ERA2-v0', entry_point='era:ERA2Env', max_episode_steps=100)  # will need to change 
register(id='ERA3-v0', entry_point='era:ERA3Env', max_episode_steps=100)  # will need to change 
register(id='ERA4-v0', entry_point='era:ERA4Env', max_episode_steps=100)  # will need to change 
register(id='ERA5-v0', entry_point='era:ERA5Env', max_episode_steps=100)  # will need to change 
register(id='ERA6-v0', entry_point='era:ERA6Env', max_episode_steps=100)  # will need to change 
register(id='ERA7-v0', entry_point='era:ERA7Env', max_episode_steps=100)  # will need to change 
register(id='ERA8-v0', entry_point='era:ERA8Env', max_episode_steps=100)  # will need to change 
register(id='ERA10-v0', entry_point='era:ERA10Env', max_episode_steps=100)  # will need to change 
register(id='ERA12-v0', entry_point='era:ERA12Env', max_episode_steps=100)  # will need to change 
register(id='ERA16-v0', entry_point='era:ERA16Env', max_episode_steps=100)  # will need to change 





