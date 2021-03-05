import gym
from gym import spaces
from PacMan.envs.mapping import Map
import numpy as np
import pygame

class PacManEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PacManEnv , self).__init__()
        self.map = Map("map.txt")
        self.action_space = spaces.Discrete(len(self.map.actions))
        self.observation_space = spaces.Box(low = 0 , high = len(self.map.char_to_image) , shape = (len(self.map.map) , len( self.map.map[0])) )
        self.possibleactions = self.map.get_possible_actions(self.map.pos)
        self.action_dict = dict(zip(np.arange(5), self.map.actions ))
        
    def step(self, action):
        if action in self.possibleactions:
            self.map.make_action(action)
        else:
            self.map.make_action(self.action_dict[0])
        self.map.move_ghosts()
        self.possibleactions = self.map.get_possible_actions(self.map.pos)
        return self.map.map , self.map.get_reward() , self.map.end_of_episode() , None
        
    def reset(self):
        self.map = Map("map.txt")
        
    def render(self, mode='human' ):
        self.map.show()
        for event in pygame.event.get() :
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        
    def close(self):
        self.map.close()
