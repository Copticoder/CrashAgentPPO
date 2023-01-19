from Backbone import ActorCriticBackbone
import tensorflow as tf

import numpy as np 
class Agent(tf.keras.Model):
    def __init__(self,rollout,num_actions, policy_clip=0.2, batch_size=64,n_epochs=10):
        
        super(Agent,self).__init__()
        # Create the backbone of the two models 
        self.backbone=ActorCriticBackbone()

        # Create the first actor-model with its own head
        self.actor = tf.keras.Sequential(self.backbone.layers)
        self.actor.add(tf.keras.layers.Dense(num_actions, activation='softmax'))

        # Create the second critic-model with its own head
        self.critic = tf.keras.Sequential(self.backbone.layers)
        self.critic.add(tf.keras.layers.Dense(1))
        
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002,epsilon=1e-5))
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002,epsilon=1e-5))
        print(self.actor(rollout))
        print(self.critic(rollout))





