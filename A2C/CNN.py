# This code includes the definition of the neural network we need (Actor-Critic)  :

import torch
from keras.layers import Input, Dense, Conv2D, Flatten , MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import Huber
from keras import backend as K

class CNN() :
    def __init__(self, shape , actsize , learning_rate ) :
        super(CNN, self).__init__()
        
        self.input = None
        self.convolution = None
        self.fcn = None
        
        self.action= None
        self.critic = None
        
        self.shape = shape
        self.actsize = actsize
        self.learning_rate = learning_rate
        
        self.convolution_maker()
        self.fcn_maker()
        self.action_maker()
        self.critic_maker()
        self.summariser()
       
        
    def convolution_maker(self) :
        self.input = Input(self.shape)
        # Convolving :
        conv1 = Conv2D(8, 7, activation="relu",  input_shape=self.shape)(self.input)
        mp1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(4, 5,padding="valid", activation="relu", input_shape=self.shape)(mp1)
        mp2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # Flatten :
        self.convolution= Flatten()(mp2)
        
    def fcn_maker(self) :
        if self.convolution == None :
            print("Initialise the convolution layer first")
            return
        self.fcn = Dense(512 , activation= "relu" )(self.convolution)
        
    def action_maker(self) :
        if self.fcn == None :
            print("Initialise the fcn first")
            return
        
        action = Dense(self.actsize, activation="softmax", kernel_initializer='he_uniform')(self.fcn)
        self.action = Model(inputs = self.input, outputs = action)
        self.action.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
    def critic_maker (self) :
        if self.fcn == None :
            print("Initialise the fcn first")
            return
     
        critic = Dense(1,kernel_initializer='he_uniform')(self.fcn)
        self.critic = Model(inputs = self.input, outputs = critic)
        self.critic.compile(loss=Huber(), optimizer=Adam(lr=self.learning_rate))
        
    def summariser(self) :
        print(self.action.summary())
        print(self.critic.summary())
    
    def save(self) :
        self.action.save("Model_action.h5")
        self.critic.save("Model_critic.h5")
        
    def update(self) :
        self.convolution_maker()
        self.fcn_maker()
        self.action_maker()
        self.critic_maker()
        
    def loadmodel(self, actor):
        self.action = load_model(actor, compile=False)
        

        
        
