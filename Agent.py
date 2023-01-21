from Backbone import ActorCriticBackbone
import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np 
from utils import LinearDecay
class Agent(tf.keras.Model):
    def __init__(self,num_actions,total_steps=80000000,entropy_c2=0.02 , gae_lambda=0.95,discount_gamma=0.99,policy_clip=0.2, batch_size=32,n_epochs=5):
        
        super(Agent,self).__init__()
        # Create the backbone of the two models 
        self.backbone=ActorCriticBackbone()
        # Create the first actor-model with its own head
        self.actor = tf.keras.Sequential(self.backbone.layers)
        self.actor.add(tf.keras.layers.Dense(num_actions, activation='softmax'))
        # Create the second critic-model with its own head
        self.critic = tf.keras.Sequential(self.backbone.layers)
        self.critic.add(tf.keras.layers.Dense(1))
        learning_rate=0.003
        self.gae_lambda=gae_lambda
        self.discount_gamma=discount_gamma
        self.observations,self.values,self.entropies,self.rewards,self.dones,self.log_probs,self.actions=[],[],[],[],[],[],[]
        #initialize the learning rate linear decay scheduler 
        lr_scheduler=LinearDecay(learning_rate,total_steps)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,epsilon=1e-5)
        
        self.critic.compile(optimizer=optimizer)
        self.actor.compile(optimizer=optimizer)
        # The memory for the agent to memorize the rollout of a single episode
        self.memory=[]
    
    def predict_actor_critic(self,state):

        state=tf.convert_to_tensor([state])
        # make the actor predict the state and then caluclate the action probability distribution
        probs=self.actor(state)
        distribution=tfp.distributions.Categorical(probs=probs) 
        #sample an action from this distribution 
        action=distribution.sample()
        #calculate the entropy in this probability distribution to account for better exploration
        entropy=distribution.entropy()
        # take the log of the probability of the chosen action for the ratio of old to new policy
        log_prob = distribution.log_prob(action)
        # get the value of this state for advantage estimation
        value = self.critic(state)
        
        action=action.numpy()[0]
        log_prob=log_prob.numpy()[0]
        value=value.numpy()[0]
        return action,log_prob,entropy,value


    def calculate_GAE(self):
        advantages=np.zeros(self.rewards.shape)
        for t in range(len(self.rewards)-1):
            multiplier=1
            for g in range(t,len(self.rewards)-1):
                delta = self.rewards[g]+(self.discount_gamma*self.values[g+1])*(1-self.dones[g]) - self.values[g]
                delta*=multiplier
                advantages[t]+=delta
                multiplier*=self.gae_lambda*self.discount_gamma
        return advantages


    def learn(self):
        """Where the agent learns from the rollouts, specifically calculating
         generalized advantage estimation, loss functions for policy, value and entropy loss, """
        self.observations,self.actions,self.log_probs,self.entropies,self.values,self.rewards,self.dones=np.array(self.observations),\
                                                                                          np.array(self.actions),\
                                                                                          np.array(self.log_probs),\
                                                                                          np.array(self.entropies),\
                                                                                          np.array(self.values),\
                                                                                          np.array(self.rewards),\
                                                                                          np.array(self.dones)
        advantages=self.calculate_GAE()
        print(advantages)

        # calculate the GAE.

    def store_rollout(self,observation,action,log_prob,entropy,value,reward,done):
        """Store the rollout of the episodes"""
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear the memory after one episode of on-policy updates"""
        self.observations,self.values,self.rewards,self.dones,self.log_probs,self.actions=[],[],[],[],[],[]
