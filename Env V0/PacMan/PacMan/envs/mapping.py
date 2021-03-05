"""
Bakr Ouairem : Map Class
"""

import pygame
import random
import numpy as np

class Map():
    def __init__(self, map ) :
        self.map = None
        self.file = map
        self.char_to_image = {'.':'sprites/dot.png' , '*':'sprites/wall.png' , 'p':'sprites/pacman.png' , 'g':'sprites/ghost.png'}
        self.Blocksize = 32
        self.WINDOW_HEIGHT = None
        self.WINDOW_WIDTH = None
        self.max = None
        self.pos = None
        self.ghost_poses = None
        self.actions = ['N', 'U','D','L','R'] # none ,up, down, left, right
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('arial', 20, bold = True )
        self.map_import()
        self.init_food_window()
        self.init_pacman()
        self.init_ghosts()
        #self.test()
    
    def map_import(self):
        self.map = []
        with open(self.file) as file :
            for line in file :
                if line[-1] == "\n" :
                    self.map.append(list(line)[:-1])
                else :
                    self.map.append(list(line))
        
    def init_food_window(self):
        c = 0
        for k,ele in enumerate(self.map):
            for i,block in enumerate(ele) :
                if block != "*":
                    self.map[k][i] = "."
                    c += 1
        self.WINDOW_WIDTH = len(self.map[0])*self.Blocksize
        self.WINDOW_HEIGHT = len(self.map)*self.Blocksize
        self.max = c
    
    def init_pacman(self):
        self.map[-2][-2] = "p"
        self.pos = ( len(self.map) - 2 , len(self.map[0]) - 2 )
        
    def get_state(self) :
        return self.map
    
    def get_reward(self):
        return self.max - sum([ i.count(".") for i in self.map])- sum([ i.count(".") for i in self.previous]) - 1
        
    def get_possible_actions(self , pos ) :
        posact = ["N"]
        if self.map[pos[0]-1][pos[1]] not in ['*','g'] :
            posact.append("U")
        if self.map[pos[0]+1][pos[1]] not in ['*','g'] :
            posact.append("D")
        if (pos[1] == 0) :
            posact.append("L")
        elif self.map[pos[0]][pos[1]-1] not in ['*','g'] :
            posact.append("L")
        if pos[1] == len(self.map[1]) -1 :
            posact.append("R")
        elif self.map[pos[0]][pos[1]+1] not in ['*','g'] :
            posact.append("R")
        return posact
        
    def make_action(self , action) : # assumes only possible actions for pacman ( pos = self.pos)
        if action == "U":
            self.map[self.pos[0]][self.pos[1]] = " "
            self.map[self.pos[0]-1][self.pos[1]] = "p"
            self.pos = ( self.pos[0]-1,self.pos[1]  )
        if action == "D":
            self.map[self.pos[0]][self.pos[1]] = " "
            self.map[self.pos[0]+1][self.pos[1]] = "p"
            self.pos = ( self.pos[0]+1,self.pos[1]  )
        if action == "L":
            self.map[self.pos[0]][self.pos[1]] = " "
            if self.pos[1] == 0 :
                self.map[self.pos[0]][len(self.map[1]) -1] = "p"
                self.pos = ( self.pos[0],len(self.map[1]) -1 )
            else :
                self.map[self.pos[0]][self.pos[1]-1] = "p"
                self.pos = ( self.pos[0],self.pos[1] - 1 )
        if action == "R":
            self.map[self.pos[0]][self.pos[1]] = " "
            if self.pos[1] == len(self.map[1]) -1 :
                self.map[self.pos[0]][0] = "p"
                self.pos = ( self.pos[0],0 )
            else :
                self.map[self.pos[0]][self.pos[1]+1] = "p"
                self.pos = ( self.pos[0],self.pos[1] + 1 )
                
    def init_ghosts(self):
        self.ghost_poses = [(13,11),(15,11),(13,16),(15,16)]
        for i in range(4):
            self.map[self.ghost_poses[i][0]][self.ghost_poses[i][1]] = 'g'
        self.previous = [".",".",".","."]
        self.prev_action = [None, None , None , None ]
    
    def get_weights(self,actions, i ):
        if len(actions) == 1 :
            return [1]
        weights = [0]
        pos = self.ghost_poses[i]
        if (pos[0]-1 , pos[1]) == self.pos :
            weights = [0,1,0,0,0]
        elif (pos[0]+1 , pos[1]) == self.pos :
            weights = [0,0,1,0,0]
        elif (pos[0], pos[1] - 1) == self.pos or (pos[0], len(self.map[0])-1) == self.pos :
            weights = [0,0,0,1,0]
        elif (pos[0], pos[1] + 1) == self.pos or (pos[0], 0) == self.pos :
            weights = [0,0,0,0,1]
        else :
            uidist = 1/((self.pos[0] - (pos[0]-1) )**2 + (self.pos[1] - (pos[1]) )**2 )
            didist = 1/((self.pos[0] - (pos[0]+1) )**2 + (self.pos[1] - (pos[1]) )**2 )
            lidist = 0
            if pos[1] - 1 < 0 :
                lidist = 1/((self.pos[0] - pos[0] )**2 + (self.pos[1] - (len(self.map[0])-1) )**2 )
            else :
                lidist = 1/((self.pos[0] - (pos[0]) )**2 + (self.pos[1] - (pos[1]-1) )**2 )
            ridist = 0
            if pos[1] + 1 > len(self.map[0])-1 :
                ridist = 1/((self.pos[0] - pos[0] )**2 + (self.pos[1] )**2 )
            else :
                ridist = 1/((self.pos[0] - (pos[0]) )**2 + (self.pos[1] - (pos[1]+1) )**2 )
            if "U" in actions :
                weights.append(uidist)
            if "D" in actions :
                weights.append(didist)
            if "L" in actions:
                weights.append(lidist)
            if "R" in actions :
                weights.append(ridist)
            return list(np.array(weights) / np.sum(weights))
        
    
    def move_ghosts(self):
        for i in range(4):
            pos = self.ghost_poses[i]
            val = self.previous[i]
            actions = self.get_possible_actions(self.ghost_poses[i])
           
            if self.prev_action[i] in actions and self.prev_action[i] != "N" :
                action = self.prev_action[i]
            else :
                action = random.choices(actions , weights = self.get_weights(actions, i))[0]
          
                
            # To add a better AI
            self.prev_action[i] = action
            if action == "U" :
                self.map[pos[0]][pos[1]] = val
                self.previous[i] = self.map[pos[0]-1][pos[1]]
                self.map[pos[0]-1][pos[1]] = "g"
                self.ghost_poses[i] = (pos[0]-1 , pos[1])
            elif action == "D" :
                self.map[pos[0]][pos[1]] = val
                self.previous[i] = self.map[pos[0]+1][pos[1]]
                self.map[pos[0]+1][pos[1]] = "g"
                self.ghost_poses[i] = (pos[0]+1 , pos[1])
            elif action == "L" :
                self.map[pos[0]][pos[1]] = val
                if pos[1] == 0 :
                    self.previous[i] = self.map[pos[0]][-1]
                    self.map[pos[0]][-1] = "g"
                    self.ghost_poses[i] = (pos[0] , len(self.map[0]) - 1 )
                else :
                    self.previous[i] = self.map[pos[0]][pos[1]-1]
                    self.map[pos[0]][pos[1]-1] = "g"
                    self.ghost_poses[i] = (pos[0] , pos[1]-1 )
            elif action == "R" :
                self.map[pos[0]][pos[1]] = val
                if pos[1] == len(self.map[0]) - 1 :
                    self.previous[i] = self.map[pos[0]][0]
                    self.map[pos[0]][0] = "g"
                    self.ghost_poses[i] = (pos[0] , 0 )
                else :
                    self.previous[i] = self.map[pos[0]][pos[1]+1]
                    self.map[pos[0]][pos[1]+1] = "g"
                    self.ghost_poses[i] = (pos[0] , pos[1]+1 )
    
    def end_of_episode(self):
        if self.max == self.get_reward() + 1 :
            return True
        for ele in self.map :
            if "p" in ele :
                return False
        return True
            
    def show(self):
        pygame.display.set_caption("Pac-Man")
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH , self.WINDOW_HEIGHT))
        background = pygame.Surface(self.screen.get_size())
        background = background.convert()
        background.fill((0, 0, 0))
        self.screen.blit(background, (0, 0))
        for y, row in enumerate(self.map):
            for x, block in enumerate(row) :
                image = self.char_to_image.get(block, None)
                if image:
                    pic = pygame.image.load(self.char_to_image[block])
                    self.screen.blit(pic , (x*self.Blocksize , y*self.Blocksize))
        text = self.font.render('Reward :'+str(self.get_reward()) , False , (250,250,250))
        self.screen.blit(text , (10,10) )
                
    def test(self) :
        self.show()
        while not self.end_of_episode() :
            actions = self.get_possible_actions(self.pos)
            action = random.choice(actions)
            self.make_action(action)
            self.move_ghosts()
            self.show()
            for event in pygame.event.get() :
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
    def close(self) :
        pygame.quit()
        quit() 

Map("map.txt")
