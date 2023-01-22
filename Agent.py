from Backbone import ActorCriticBackbone
import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np 
from utils import LinearDecay
class Agent(tf.keras.Model):
    def __init__(self,num_actions,rollout_horizon=1024,total_steps=80000000,entropy_c2=0.02,vf_c1=1 , gae_lambda=0.95,discount_gamma=0.99,epsilon_clip=0.2, batch_size=32,n_epochs=5):
        super(Agent,self).__init__()
        # Create the backbone of the two models 
        self.backbone=ActorCriticBackbone()
        # Create the first actor-model with its own head
        self.actor = tf.keras.Sequential(self.backbone.layers)
        self.actor.add(tf.keras.layers.Dense(num_actions, activation='softmax',kernel_initializer=tf.initializers.Constant(value=0.01)))
        # Create the second critic-model with its own head
        self.critic = tf.keras.Sequential(self.backbone.layers)
        self.critic.add(tf.keras.layers.Dense(1))
        learning_rate=0.003
        self.entropy_c2=entropy_c2
        self.vf_c1=vf_c1
        self.n_epochs=n_epochs
        self.rollout_horizon=rollout_horizon
        self.gae_lambda=gae_lambda
        self.discount_gamma=discount_gamma
        self.epsilon_clip=epsilon_clip
        # Initialize agent memory 
        self.observations,self.values,self.rewards,self.dones,self.log_probs,self.actions=[],[],[],[],[],[]
        #initialize the learning rate linear decay scheduler 
        lr_scheduler=LinearDecay(learning_rate,total_steps)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,epsilon=1e-5)
        
        self.critic.compile(optimizer=self.optimizer)
        self.actor.compile(optimizer=self.optimizer)
        #checkpoint objects for saving the models for actor and critic
        self.actor_checkpt = tf.train.Checkpoint(self.actor)
        self.critic_checkpt = tf.train.Checkpoint(self.critic)
    def predict_actor_critic(self,state):
        state=tf.convert_to_tensor([state])
        #normalize the frames 
        state/=255

        # make the actor predict the state and then caluclate the action probability distribution
        probs=self.actor(state)
        distribution=tfp.distributions.Categorical(probs) 
        #sample an action from this distribution 
        action=distribution.sample()
        #calculate the entropy in this probability distribution to account for better exploration
        entropy=distribution.entropy()
        # take the log of the probability of the chosen action for the ratio of old to new policy
        log_prob = distribution.log_prob(action)
        # get the value of this state for advantage estimation
        value = self.critic(state)
        return np.array(action),np.array(log_prob,dtype=np.float32),np.array(entropy,dtype=np.float32),np.array(value,dtype=np.float32)


    def learn(self):
        """Where the agent learns from the rollouts, specifically calculating
         generalized advantage estimation, loss functions for policy, value and entropy loss, """
        self.observations,self.actions,self.log_probs,self.values,self.rewards,self.dones=np.array(self.observations,dtype=np.float32),\
                                                                                          np.array(self.actions,dtype=np.float32),\
                                                                                          np.array(self.log_probs,dtype=np.float32),\
                                                                                          np.array(self.values,dtype=np.float32),\
                                                                                          np.array(self.rewards,dtype=np.float32),\
                                                                                          np.array(self.dones,dtype=np.float32)
        
        """This is for calculating the advantage which is the accumulation of rewards
        and state values according to the equations in the PPO paper: https://arxiv.org/pdf/1707.06347.pdf
        -> At = δt + (γλ)δt+1 + · · · """
        advantages=np.zeros(self.rewards.shape,dtype=np.float32)
        for t in range(len(self.rewards)-1):
            multiplier=1
            for g in range(t,len(self.rewards)-1):
                delta = self.rewards[g]+(self.discount_gamma*self.values[g+1])*(1-self.dones[g]) - self.values[g]
                delta*=multiplier
                advantages[t]+=delta
                multiplier*=self.gae_lambda*self.discount_gamma
        
        for _ in range(self.n_epochs):
            
            # Shuffle the data for minibatches 
            num_samples = self.rewards.shape[0]
            shuffled_indices=np.random.permutation(num_samples)
            
            self.observations,self.actions,self.log_probs,self.values,self.rewards,self.dones=self.observations[shuffled_indices],\
                                                                                                            self.actions[shuffled_indices],\
                                                                                                            self.log_probs[shuffled_indices],\
                                                                                                            self.values[shuffled_indices],\
                                                                                                            self.rewards[shuffled_indices],\
                                                                                                            self.dones[shuffled_indices]
            
            number_of_batches = len(self.rewards)//32
            for increment in range(0,number_of_batches,1):
                with tf.GradientTape() as tape:
                    #Take Batches from the rollout 
                    old_log_probs=self.log_probs[increment*32:(increment+1)*32]
                    states=self.observations[increment*32:(increment+1)*32]/255
                    actions=self.actions[increment*32:(increment+1)*32]
                    #convert them to tensorflow tensors for the gradient tape and backprop procedure
                    old_log_probs=tf.convert_to_tensor(old_log_probs)
                    states=tf.convert_to_tensor(states)
                    actions=tf.convert_to_tensor(actions)
                    actions=tf.squeeze(actions,1)
                    #do inference for the new log_probabilites for the ratio between old policy and new policy
                    probs=self.actor(states)
                    distribution=tfp.distributions.Categorical(probs) 
                    entropies=distribution.entropy() #calculate entropies for the policy probability distribution. That enhances exploration
                    log_probs=distribution.log_prob(actions)
                    values = self.critic(states)
                    
                    m_advantages=advantages[increment*32:(increment+1)*32]
                    old_values=self.values[increment*32:(increment+1)*32]
                    
                    log_probs=tf.expand_dims(log_probs,1)
                    m_advantages=tf.expand_dims(m_advantages,1)
                    old_values=tf.squeeze(old_values,2)
                    ratio_log_probs=tf.math.exp(log_probs-old_log_probs)
                    #policy losses according to the ppo paper
                    pg_l1=-ratio_log_probs*m_advantages
                    pg_l2=-m_advantages*tf.clip_by_value(ratio_log_probs, clip_value_min=1-self.epsilon_clip, clip_value_max=1+self.epsilon_clip)
                    #the overall loss of the actor
                    actor_loss=tf.math.reduce_mean(tf.math.maximum(pg_l1,pg_l2))
                    
                    # normalize the advantages to have 0 mean and 1 std 
                    returns = m_advantages + old_values
                    returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)
                    # The critic losses with clipping 
                    v_loss1=tf.math.square(values-returns)
                    clipped_v=tf.clip_by_value(values, clip_value_min=1-self.epsilon_clip, clip_value_max=1+self.epsilon_clip)
                    v_loss2=tf.math.square(clipped_v-returns)
                    critic_loss=tf.math.maximum(v_loss1,v_loss2)
                    critic_loss=tf.math.reduce_mean(critic_loss)
                    #take the mean
                    entropies=tf.math.reduce_mean(entropies)
                    # the total loss including the actor, the critic the entropies 
                    total_loss=actor_loss - entropies*self.entropy_c2 + critic_loss*self.vf_c1
                # Compute gradients
                grads = tape.gradient(total_loss, self.backbone.trainable_variables + self.actor.trainable_variables + self.critic.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_variables + self.actor.trainable_variables + self.critic.trainable_variables))
        self.clear_memory()

    def store_rollout(self,observation,action,log_prob,value,reward,done):
        """Store the rollout of the episodes"""
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def save_models(self,episode):
        """A method for saving the actor critic models checkpoints"""
        self.actor_checkpt.save(f'./checkpoints/actor/{episode}/')
        self.critic_checkpt.save(f'./checkpoints/critic/{episode}/')
    
    def restore_models(self,episode):
        """a method for restoring the models from checkpoints"""
        self.actor_checkpt.restore("./checkpoints/actor/{episode}/")
        self.critic_checkpt.restore("./checkpoints/critic/{episode}/")

    def clear_memory(self):
        """Clear the memory after one episode of on-policy updates"""
        self.observations,self.values,self.rewards,self.dones,self.log_probs,self.actions=[],[],[],[],[],[]