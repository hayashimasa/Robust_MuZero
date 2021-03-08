# This code includes the definition of the agent (Main file)   :

from CNN import CNN
from Breakout import Breakout
import matplotlib.pyplot as plt
import time as tm
import numpy as np


class Agent():
    def __init__(self , lr = 0.001 , number_episodes = 5000 , epochs = 1 ) :
        self.br = Breakout()
        self.lr = lr
        self.brain = CNN(self.br.data_shape , self.br.action_size , lr)
        self.ep = number_episodes
        self.max_score = -1
        self.scores = []
        self.avscores = []
        self.maxscores = []
        self.tolerance_proba = [0]
        self.comp = 0
        self.stopping = 500
        self.epochs = epochs
        self.rand_NN_thres = 100
        self.actor = None
        self.critic = None
        self.test_episodes = 100
        self.previous_averaging = 20
        self.vis_pres = 1
        
    def average(self) :
        return np.mean(self.scores[ len(self.scores) - min(len(self.scores) ,  self.previous_averaging)  : ])
        
    def discounted_rewards(self, reward) : # Defines the q function
        gamma = 0.99
        running_add = 0
        discounted = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))) :
            if reward[i] != 0 :
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted[i] = running_add
        discounted -= np.mean(discounted)
        if np.std(discounted)> 0 :
            discounted /= np.std(discounted)
        return discounted
        

    def run(self) : # The running function
        T1 = tm.time()
        T2 = 0
        for e in range(self.ep) :
            # Reset the state :
            state = self.br.reset()
            
            # Training loop , either we go over a certain number of steps, or done
            done = False
            i = 0
            j = 0
            score = 0
            while not done and not i > self.stopping and not j > self.stopping/5 :
                i += 1
                if e%self.vis_pres == 0 :
                    self.br.env.render()
                
                # The randomiser :
                if e<= self.rand_NN_thres :
                    action = self.br.env.action_space.sample()
                    next_state , reward, done , info = self.br.env.step(action)
                # We let the agent go itself :
                if e > self.rand_NN_thres:
                    action_soft = self.brain.action.predict(state.reshape(1, state.shape[0] ,state.shape[1], state.shape[2] ))[0]
                    action , next_state , reward, done , info = self.br.step(action_soft) # take the most likely action to work
                    if action != 1 :
                        j += 1
                    if action == 1 :
                        j = 0
                self.br.history(state, action , reward)
                #print(action)
                state = next_state
                score += reward
            
            self.scores.append(score)
            average = self.average()
            save = average > self.max_score and e > self.rand_NN_thres
            if save :
                self.max_score = average
                self.brain.save()
            
            self.avscores.append(average)
            self.maxscores.append(self.max_score)
            T2 = tm.time()
            print( "Episode: {}/{}, Final Score: {}, Average: {:.2f}, Naturally Ended: {}, Updated : {} , Done in : {} ".format(e, self.ep, score, self.average(), done, save , T2 - T1 ) )
            T1 = tm.time()
            self.update_nn_and_restart()
                
    def update_nn_and_restart(self) :
        self.comp += sum(self.br.rew_hist)
        if  self.comp == 0 :  # New Neural Network
            self.brain.update()
        elif self.comp > 0 and sum(self.br.rew_hist) == 0 :
            r = np.random.rand()
            if  r > np.mean(self.tolerance_proba) and r < 0.9 :
                self.brain.update()
                print("Updated")
            else :
                self.tolerance_proba.append(1)
                # Learn the data
                self.learn()
        else :
            self.tolerance_proba.append(1)
            # Learn the data
            self.learn()
        
        
    def learn(self) :
        states = self.br.state_hist
        actions = self.br.act_hist
        rewards = self.br.rew_hist
        
        crit_val = self.brain.critic.predict(np.vstack(states))[:,0]
        discounted_rew = self.discounted_rewards(rewards)
        # print(discounted_rew)
        
        self.brain.action.fit(np.vstack(states) , np.vstack(actions), batch_size= 32, sample_weight = discounted_rew - crit_val  , epochs = self.epochs , verbose = 0 )
        
        self.brain.critic.fit(np.vstack(states), discounted_rew , batch_size= 32, epochs = self.epochs , verbose = 0 )
        
        self.br.state_hist = []
        self.br.act_hist = []
        self.br.rew_hist = []
        
    def test_a_neural_network(self, actor , critic ):
        self.actor = self.brain.loadmodel(actor)
        if critic != '' :
            self.critic = load_model(critic, compile=False)
        for e in range(self.test_episodes) :
            state = self.br.reset()
            done = False
            score = 0
            while not done :
                self.br.env.render()
                action_soft = self.brain.action.predict(state.reshape(1, state.shape[0] ,state.shape[1], state.shape[2] ))[0]
                #print(np.argmax(action_soft)) 
                action , next_state , reward, done , info = self.br.step(action_soft)
                score += reward
            print("Episode: {}/{}, Final Score: {}".format(e, self.ep, score))
            
    def visualise(self) :
        plt.plot(self.scores)
        plt.plot(self.avscores)
        plt.plot(self.maxscores)
        plt.show()
            
ag = Agent()
#ag.run()
ag.test_a_neural_network("Model_action.h5", "")
#ag.visualise()
