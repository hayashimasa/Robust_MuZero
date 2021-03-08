# This code includes all important functions to initialise the problem itself  :

import gym , cv2
import numpy as np

class Breakout() :
    def __init__(self, env = "Breakout-v0" ) :
        super().__init__()
        self.envname = env
        self.env = None
        self.action_size = None
        self.data_shape = None
        # History of states, actions and rewards
        self.state_hist = []
        self.act_hist = []
        self.rew_hist = []
        self.image_memory = None
        
        self.initenv()
        self.getactsize()
        self.getobsize()
        
    def initenv(self):
        self.env = gym.make(self.envname)
    
    def getactsize(self):
        if self.env == None :
            print("env not initialised correctly ")
            return
        self.action_size = self.env.action_space.n
        
    def getobsize(self):
        if self.env == None :
            print("env not initialised correctly ")
            return
        self.data_shape = self.env.observation_space.shape
        self.image_memory = np.zeros(self.data_shape)
    
    def history(self, state, action , reward) :
        self.state_hist.append(state.reshape(1, state.shape[0] ,state.shape[1], state.shape[2]) )
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.act_hist.append(action_onehot)
        self.rew_hist.append(reward)
        
    def reset(self) :
        state = self.env.reset()
        return state
        
    def step(self,  actions) :
        action = np.random.choice(self.action_size, p=actions)
        next_state , reward , done , info = self.env.step(action)
        return action , next_state , reward, done , info
        
        
        
