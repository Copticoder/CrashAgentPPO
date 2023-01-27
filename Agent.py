import numpy as np 
import torch 
from networks import ActorCritic
import os
class Agent(torch.nn.Module):
    def __init__(self,num_actions,learning_rate,rollout_horizon=2048,entropy_c2=0.01,vf_c1=1 , gae_lambda=0.95,discount_gamma=0.99,epsilon_clip=0.2, batch_size=32,n_epochs=5):
        super(Agent,self).__init__()
        self.use_cuda = torch.cuda.is_available() # Autodetect CUDA
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print('Device:', self.device)
        # Create the whole model for both the actor and the critic 
        self.model = ActorCritic(num_actions).to(self.device) # save the model, Tensor.to(device) Moves and/or casts the parameters and buffers.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) # Implements Adam algorithm.
        #create a state dict for saving the model
        self.state_dict = self.model.state_dict()
        self.batch_size=batch_size
        self.entropy_c2=entropy_c2
        self.vf_c1=vf_c1
        self.n_epochs=n_epochs
        self.rollout_horizon=rollout_horizon
        self.gae_lambda=gae_lambda
        self.discount_gamma=discount_gamma
        self.epsilon_clip=epsilon_clip
        self.learning_rate=learning_rate
        # Initialize agent memory 
        self.observations,self.values,self.rewards,self.dones,self.log_probs,self.actions=[],[],[],[],[],[]        

    def learn(self):
        """Where the agent learns from the rollouts, specifically calculating
         generalized advantage estimation, loss functions for policy, value and entropy loss, """

        gae = 0 # first gae always to 0
        returns = []
        """This is for calculating the advantage which is the accumulation of rewards
        and state values according to the equations in the PPO paper: https://arxiv.org/pdf/1707.06347.pdf
        -> At = δt + (γλ)δt+1 + · · · """
        for step in reversed(range(len(self.rewards))): # for each positions with respect to the result of the action 
            delta = self.rewards[step] + self.discount_gamma * self.values[step + 1] * (1-self.dones[step]) - self.values[step] 
            gae = delta + self.discount_gamma * self.gae_lambda * (1-self.dones[step]) * gae 
            returns.insert(0, gae + self.values[step])
        
        self.values=self.values[:-2]
        returns = torch.cat(returns).detach() # concatenates along existing dimension and detach the tensor from the network graph, making the tensor no gradient
        self.log_probs = torch.cat(self.log_probs).detach() 
        self.values = torch.cat(self.values).detach()
        self.observations = torch.stack(self.observations)
        self.actions = torch.cat(self.actions)
        advantages = returns - self.values # compute advantage for each action
        dataset = torch.utils.data.TensorDataset(self.log_probs, self.observations, self.actions,returns, advantages)
        for _ in range(self.n_epochs):
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for minibatch in data_loader:
                log_probs_batch, observations_batch, actions_batch,returns_batch,advantages_batch=minibatch
                #normalize advantages on batch level
                advantages_batch-=advantages_batch.mean()
                advantages_batch/=(advantages_batch.std()+1e-8)
                dist, values = self.model(observations_batch)

                entropy = dist.entropy().mean()
                actions_batch = actions_batch.reshape(1, len(actions_batch)) # take the relative action and take the column
                new_log_probs = dist.log_prob(actions_batch)
                new_log_probs = new_log_probs.reshape(new_log_probs.shape[1],1)
                ratio = (new_log_probs - log_probs_batch).exp() # new_prob/old_prob
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages_batch 
                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (returns_batch - values).pow(2).mean()
                loss = self.vf_c1 * critic_loss + actor_loss - self.entropy_c2 * entropy
                self.optimizer.zero_grad() # in PyTorch, we need to set the gradients to zero before applying gradient steps.
                loss.backward() 
                self.optimizer.step()             
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
        # Create the directory if it doesn't exist
        file_path=f'./checkpoints/actor-critic/{episode}/model.pth'
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        """A method for saving the actor critic models checkpoints"""
        # Save the state_dict to a file
        torch.save(self.state_dict, file_path)
    
    def restore_models(self,episode):
        """A method for loading the models"""
        state_dict = torch.load(f'./checkpoints/actor-critic/{episode}/model.pth')
        # Load the state_dict into the model
        self.model.load_state_dict(state_dict)

    def clear_memory(self):
        """Clear the memory after one episode of on-policy updates"""
        self.observations,self.values,self.rewards,self.dones,self.log_probs,self.actions=[],[],[],[],[],[]