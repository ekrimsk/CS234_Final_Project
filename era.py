import numpy as np

import gym  # so it can be super class 
from gym import utils

import matplotlib.pyplot as plt 
import os

# EREZ ADD 
from gym import error, spaces # we ar going to try to overwrite mujoco non-sense 
from utils.era_utils import binvec
from era_params import get_era_params
# Cart Pole is Good Resource 
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py


# whats best way to load in all the parameters
# create seperate file with set of parameters for each n 

# TODO -- add option for spring slips in dynamics (dont need to implement just yet though)

# NOTE Need to remove mujoco inheritance, will lead to issues 

#class ERAEnv(mujoco_env.MujocoEnv, utils.EzPickle):
class ERAEnv(gym.Env, utils.EzPickle):

    """
    Description:


    Observation:
        Type: Box(2n + 3)
        Num             Observation                             Min         Max
        0 -- (n-1)      Corresponding clutch state              0           1
        n -- (2n - 1)   Corresponding spring position      
        2n              Current angle (output position)
        2n + 1          Current angular velocity                -Inf        Inf 
        2n + 2          Desired Target

        Will set limit slightly high so we can observe failure and terminate for
        Spring displacement limits 
    
        NOTE:  Could make desired target an array that we step through so that this 
        is tracking and NOT just step inputs.
        This would mean a larger state space but that might be ok 


    Actions:
        Type: Discrete - 0 to (2^n - 1)
            Corresponding action based off "bit" vector of discrete input
            0       Spring Clutch Engaged to Frame
            1       Spring Clutch Engaged to Output 


    Rewards:

    Starting State:


    Episode Termination Conditions:

    Useful resources:
        Addings args in initializer:  https://stackoverflow.com/questions/5169257/init-and-arguments-in-python

    """

    # NOTE -- want to move this into init to set framerate consistently but issues with monitor !!
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }



    # How to add args to inializer
    def __init__(self, num=None):
        # init EzPickle for saving model 
        utils.EzPickle.__init__(self)

        # Physics Definitions 
        self.gravity = 9.81
        self.mass_ball = 5              # kg to lift, ignoring mass of arm        
        self.length_arm = 0.35          # m, approximate forearm length 
        self.J =  self.mass_ball * (self.length_arm**2)     # rotational inertia


        # Solver Options 
        self.kinematics_integrator = 'euler'    # gives us an option to switch to RK4 later 
        #self.dt = 0.02                          # time between state updates in integrator 
        self.dt = 0.01                          # time between state updates in integrator  

        self.control_freq = 20.0 # Hz 


        # Copied from cartpole -- for rendering
        # change control freq to 20 
        #metadata = {
        #    'render.modes': ['human', 'rgb_array'],
        #    'video.frames_per_second' : np.round(self.control_freq) 
        #}


        self.frame_skip = np.int(1/(self.control_freq * self.dt))


        # Define Arm Minimimum and Maximimum angles 
        self.angle_max = np.deg2rad(45)
        self.angle_min = np.deg2rad(-85)   # NOTE may want to raise a little 

        self.lock_window = np.deg2rad(4)    # for locking the actuator
        self.lock_vel = 0.05                # rad/s -- play with this 

        # NOTE: To make more organized might want self.act_params to store things like  stiffness + lengths? 


        assert ( 2 * self.lock_window*self.control_freq >= self.lock_vel)    # otherwise could swing through the window undetected 

        self.viewer = None
        self.state = None

        # how to have default params??? 

        if (num==None):
            num=1

        n = num
        self.n = n     


        #---------------- Action Space Defintion --------------------------
        num_actions = pow(2, n)
        self.action_space = spaces.Discrete(num_actions)

        # Action Table -- For speed can call this later 
        self.action_table = np.zeros((num_actions, n))
        for ii in range(num_actions):
            self.action_table[ii,:] = binvec(ii, n)     # call from utils now 

        #-------- Load in number of springs specific actuator parameters ----------
        params = get_era_params(n)
        self.lims_low = params.lims_low         
        self.lims_high = params.lims_high
        self.r = params.r              # wrap radii , TODO pull from config 
        self.k = params.k       #   TODO -- pull from config file 
        self.b = params.b


        # Observation space def
        obs_high =  np.concatenate((np.ones(n),
                                    self.lims_high,
                                    [np.pi/2,      # current angle 
                                    np.Inf,          # velocity
                                    np.pi/2]      
                                    ))
        #print('obs high: ', obs_high)
                                
        obs_low =  np.concatenate((np.zeros(n),
                                    self.lims_low,
                                    [-np.pi/2,      # current angle 
                                    -np.Inf,        # velocity
                                    -np.pi/2]       # goal angle        
                                    ))

        #self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32) # the clutch state observations could be bool
        self.observation_space = spaces.Box(obs_low, obs_high) # the clutch state observations could be bool


        # NOTE Add seeed for randomization? (see gym/cartpole.py )
        # Some Debug Print Statements 
        # self.summary()


    def step(self, a, do_update=True, verbose=False):

        assert self.action_space.contains(a), "%r (%s) invalid"%(a, type(a))
        
        # NOTE: could add a control cost or other features to the reward functio if we want  
        # NOTE: Could create seperate "cost function" that we call for modularity  
        n = self.n

        # TODO NOW STATE WILL NEED RO INCLUDE LOCKED POSITION -- STILL HAVE GOAL AT END 
        state = self.state      # clutch states, spring lengths, end effector position, velocity 
        clutches = state[0:n] 
        x = state[n:(2*n)]
        theta = state[2*n]
        theta_dot = state[2*n + 1]

        #u = binvec(a, n)    # TODO -- make sure to move this to an era_utils module
        u = self.action_table[a, :]
        done = False   
        done_reason = None

        # TODO -- potentially add in accounting for slip based off change in clutch states 
        for ii in range(self.frame_skip):
            if self.kinematics_integrator == 'euler':
                F = u * ((x * self.k) + self.b)
                Tau_grav = self.gravity * self.mass_ball * self.length_arm * np.cos(theta) 
                Tau_act = np.dot(self.r, F)
                theta_ddot = (Tau_act - Tau_grav)/self.J
                theta = theta + (self.dt * theta_dot)
                x = x - self.dt * u * (self.r * theta_dot)
                theta_dot = theta_dot + self.dt*theta_ddot
            else:
                # NOT ACCOUNTED FOR YET
                assert False 
                     
        # trying quadratic penalty        
        reward_dist = -(self.goal - theta)**2
        reward = reward_dist    # may get overwritten by a condition below 


        # Check for terminal conditions 
        if (theta < self.angle_min) or (theta > self.angle_max):
            done = True
            #reward = -10
            reward = -200

            done_reason="Angle out of range"
        elif any(x > self.lims_high) or any(x < self.lims_low):
            done = True
            #reward = -10
            reward = -200

            done_reason="x out of range"

        elif ((theta < self.goal + self.lock_window) and (theta > self.goal - self.lock_window)) and (np.abs(theta_dot) < self.lock_vel):
            done = True
            # reward = 10  
            reward = 100   
            done_reason="Goal reached"
        
        final_state = np.concatenate((u,
                                     x,
                                     [theta,
                                     theta_dot]))

        ob = np.concatenate((final_state, [self.goal]))

        if do_update:
            self.state = final_state

        # Add forces and torques to dict for debug? 
        return ob, reward, done, dict(reward_dist=reward_dist, done_reason=done_reason)


    def reset(self, goal=None, state=None, verbose=False):

        # NOTE -- could cahnge these randomization to use a class specific seeding (see polecart.py)
        
        if state is None:

            # True random 
            clutches = np.random.randint(2, size=self.n)
            theta_dot_0 = np.random.uniform(low=-0.1, high=0.1)
            theta_0 = np.random.uniform(low=0.4 * self.angle_min, high=0.4 * self.angle_max)


            #x = np.random.uniform(low=0.2 * self.lims_low, high=0.2*self.lims_high)
            x = -self.r * theta_0

            # other optopm -- TODO -- as as option? 
            #clutches = np.zeros(self.n)
            #lengths = self.slack_length * 0.5 * (self.min_stretch + self.max_stretch)
            #theta_0 = 0
            #theta_dot_0 = 0

        #else:        
            # NOTE IMPLENTED BECAUSE OPENAI UNHAPPY 

        if goal is None:
            #goal = np.random.uniform(low=0.5*self.angle_min, high=0.5*self.angle_max)
            goal = np.random.uniform(low=0.8*self.angle_min, high=0.8*self.angle_max)
        # else:
                        
        self.state = np.concatenate([clutches, x, [theta_0, theta_dot_0]])
        self.goal = goal

        if verbose:
            #max_backward = -min((self.max_stretch * self.slack_length - lengths)/self.r)
            #max_forward  = min((lengths - self.min_stretch * self.slack_length)/self.r)
            print('Start State Report:')
            print('\tGoal Angle: ', self.goal)
            print('\tStarting Angle: ', theta_0)
            print('\tDesired Ang displacement: ', self.goal - theta_0)
            #print('\tMax Forward Rotation  : ', max_forward)       # from initial length limits 
            #print('\tMax backward Rotation  : ', max_backward)       # from initial length limits 

        obs = np.concatenate([self.state, [self.goal]])  
        return obs   

    def set_goal(self, new_goal):
        self.goal = new_goal

    def set_state(self, new_state):
        self.state = new_state

    def get_r(self):
        return self.r

    def get_n(self):
        return self.n

    def render(self, mode='human', close=None):
        screen_width = 600
        screen_height = 400

        world_height = 2.5 * self.length_arm
        scale = screen_height/world_height

        arm_len = scale * (self.length_arm)
        arm_width = 10.0    # idk? 

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            # https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
            self.viewer = rendering.Viewer(screen_width, screen_height)


            # TODO 
            # Add mass on end of arm visually
            # Add placeholder for target pooint


            l, r, t, b = -arm_width/2, arm_len, arm_width/2,  -arm_width/2
            arm = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])

            x_axle = screen_width/2
            y_axle = screen_height/2
            
            self.arm_trans = rendering.Transform(translation=(x_axle, y_axle))
            arm.add_attr(self.arm_trans)
            arm.set_color(.8,.6,.4)
            self.viewer.add_geom(arm)

            # Add axle
            self.axle = rendering.make_circle(arm_width/2)
            self.axle.set_color(.5,.5,.8)
            self.axle.add_attr(self.arm_trans)
            self.viewer.add_geom(self.axle)

            # Add target pos 
            self.goal_trans = rendering.Transform(translation=(x_axle + arm_len*np.cos(self.goal), 
                                                               y_axle + arm_len*np.sin(self.goal)))            
            goal = rendering.make_circle(2 * arm_width)
            goal.set_color(.9,.1,.1)
            goal.add_attr(self.goal_trans)
            self.viewer.add_geom(goal)

            self._arm_geom = arm
            self._goal_geom = goal 
            self._x_axle = x_axle
            self._y_axle = y_axle


        if self.state is None: return None

        # Edit the pole polygon vertex
        arm = self._arm_geom
        goal = self._goal_geom
        x_axle = self._x_axle
        y_axle = self._y_axle

        l, r, t, b = -arm_width/2, arm_len, arm_width/2,  -arm_width/2
    

        #if (self.state[2*self.n + 2] == 1):
        #goal.set_color(0.1, 0.1, 0.9)
        #else:
        goal.set_color(0.9, 0.1, 0.1)

        self.arm_trans.set_rotation(self.state[2*self.n])

        self.goal_trans.set_translation(x_axle + arm_len*np.cos(self.goal), 
                                        y_axle + arm_len*np.sin(self.goal))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    # Ripped from polecart.py
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



    # Print out a lot of things about the configuration
    def summary(self):
        max_static = self.gravity * self.length_arm * self.mass_ball
        print("Max. required static torque:{:6.2f} Nm".format(max_static))

        F_max = (self.lims_high*self.k) + self.b
        Tau_act_max = np.dot(self.r, F_max)
        print("Max. individual possible torques (Nm) ", F_max * self.r)

        print("Max. possible total torque:{:6.2f} Nm".format(Tau_act_max))

        F_min = (self.lims_low*self.k) + self.b
        Tau_act_min = min(F_min * self.r)    # lowest torque from the list 

        print("Min. individual possible torques (Nm) ", F_min * self.r)
        print("Min possile total non zero torque:{:6.2f} Nm".format(Tau_act_min))

        #print('Max allowable spring displacement: ', self.slack_length * (self.max_stretch - self.min_stretch))
        print('Max extension/retraction over: ', (self.angle_max - self.angle_min)*self.r)

        # Does this make a feasible rubber spring???
        #E = 300e3   
        #A = self.stiffness * self.slack_length/E;
        #w = A/(1e-3)
        #print("Assuming 1mm thickness, spring widths are (cm) ", w)

    # TODO -- add comments as to what this is 
    def act_greedy(self):        
        greedy_reward = -np.Inf   # initialize to min possible reward
        for action in range(self.action_space.n):  
            obs, rew, done, _ = self.step(action, do_update=False)
            if (rew > greedy_reward):
                greedy_reward = rew
                greedy_action = action
        return greedy_action




    def act_pd(self, p_gain=2.5, d_gain=0.9):

        n = self.n
        x = self.state[n:(2*n)]
        theta = self.state[2*n]
        theta_dot = self.state[2*n + 1]
        theta_goal = self.state[-1]

        p_error = theta_goal - theta
        d_error = -theta_dot 

        pd_target = (p_gain*p_error) + (d_gain * d_error)

        # Evaluate the resultant torque for each action 
        #torques = np.zeros((self.action_space.n,))
        pd_action = None
        min_error = np.Inf
        for action in range(self.action_space.n):  
            u = self.action_table[action, :]
            F = u * ((x * self.k) + self.b)
            Tau_grav = self.gravity * self.mass_ball * self.length_arm * np.cos(theta) 
            Tau_act = np.dot(self.r, F)
            Tau_total = (Tau_act - Tau_grav)

            if np.abs(Tau_total - pd_target) < min_error:
                # check if this will kill us!
                ob, reward, done, info = self.step(action, do_update=False)
                if (reward > -200): # if wont kill us 
                    min_error = np.abs(Tau_total - pd_target) 
                    pd_action = action
                elif (reward > 0):
                    min_error = 0
                    pd_action = action
                    break
            
        if pd_action is None:
            pd_action = self.act_greedy()

        return pd_action



    def torque_plot(self):

        #min_torques = np.zeros(self.action_space.n)
        #max_torques = np.zeros(self.action_space.n)
        n = self.n
        mid_torques = np.zeros(self.action_space.n)
        torque_range = np.zeros((2, self.action_space.n))
        for action in range(self.action_space.n):  

            x_mid = np.zeros(self.n) 
            x_low = self.lims_low
            x_high = self.lims_high 

            u = self.action_table[action, :]
            T_low = np.dot(self.r, u * ((x_low * self.k) + self.b))
            T_high = np.dot(self.r, u * ((x_high * self.k) + self.b))
            T_mid = np.dot(self.r, u * ((x_mid * self.k) + self.b))
            mid_torques[action] = T_mid
            torque_range[1, action] = T_high - T_mid
            torque_range[0, action] = T_mid - T_low



        fig = plt.figure()  # new figure for no 

        plt.errorbar(np.arange(self.action_space.n), mid_torques, torque_range, color='black', ls='--', marker='o', capsize=5, capthick=1, ecolor='black')


        # TODO -- make this look ok 

        #plt.xlabel("time")
        #plt.ylabel(r'$\theta$')
        # plt.legend(loc='best')

        #plt.xlim(0, max(t))
        plt.ylabel('Actuator Torque Range (Nm)')
        plt.xlabel('Action Number')
        plt.savefig('force_range_fig.png', bbox_inches='tight')        # make naming not dumb though 
    

"""
    Subclass Definitions for inidividual numbers of springs 
"""
class ERA1Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=1)

class ERA2Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=2)

class ERA3Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=3)

class ERA4Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=4)

class ERA5Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=5)        

class ERA6Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=6)

class ERA7Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=7)

class ERA8Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=8)        


class ERA10Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=10)       

class ERA12Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=12)       


class ERA16Env(ERAEnv):  # making a subclass 
    def __init__(self):
        ERAEnv.__init__(self, num=16)            