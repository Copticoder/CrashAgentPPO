# About CrashAgentPPO
This repo contains my implementation for a variant of the PPO reinforcement learning algorithm on the retro environments by openAI.
Specifically, on Crash Bandicoot The Huge Adventure game. This implementation and training was done on my personal laptop on a mid-range GPU Nvidia 1650 4GB memory 1485 MHz Base Clock. THIS IS ONLY FOR EDUCATIONAL PURPOSES AND IT'S A PERSONAL PROJECT.
# KeyFiles 
* Agent.py : Contans the agent class, including the agent memory, neural networks initialization, the learning algorithm (PPO) and training loop and finally saving and loading the models
* networks.py : contains the classes for the backbone (Shared CNN network), the actor head, the critic head and finally the whole ActorCritic archeticture.
* main.py : The program that runs the crash environment and begins rolling out episodes and collecting data for the agnet as well as saving the score results.

# Details
This variant of PPO doesn't contain the following original implementation details:
* didn't add vectorized environments where there are multiple environments collecting data in parallel.
* didn't add linear Learning Rate Annealing.

However, The model archeticure is as follows (From Pytorch Statedict) the input are stacked 4 frames of the game in greyscale: 
```
#Backbone CNN
Backbone.conv1.weight    torch.Size([32, 4, 3, 3])
Backbone.conv1.bias      torch.Size([32])
Backbone.conv2.weight    torch.Size([64, 32, 3, 3])
Backbone.conv2.bias      torch.Size([64])
Backbone.conv3.weight    torch.Size([64, 64, 6, 6])
Backbone.conv3.bias      torch.Size([64])
Backbone.conv4.weight    torch.Size([64, 64, 6, 6])
Backbone.conv4.bias      torch.Size([64])
Backbone.fc.weight       torch.Size([512, 26752])
Backbone.fc.bias         torch.Size([512])
#Actor 
actor.fc.weight          torch.Size([8, 512])
actor.fc.bias    torch.Size([8])
#Critic
critic.fc.weight         torch.Size([1, 512])
critic.fc.bias   torch.Size([1])
```
## Hyperparameters:
* learning_rate=3e-5
* rollout_horizon=2048
* entropy_c2=0.01
* vf_c1=1
* gae_lambda=0.95
* discount_gamma=0.99
* epsilon_clip=0.2
* batch_size=64
* n_epochs=10

The agent learned over the course of 1200 episodes equivalently 8 hours of training with approximately 2 Million steps.

PPO Paper: https://arxiv.org/pdf/1707.06347.pdf

## Reward Reshaping
The reward function in the LUA program file is as follows, when crash moves forward it gets 0.03 reward and -0.03 if backwards. If it eats an apple, hits a box, or hits an enemy it gets a reward of 1. If it died and lost a life gets -1 punishment. 

# Results
The Overall results are noisy which is the state of the majority of RL algorithms. However, as shown some episodes show high rewards and others very low with an average of +75.

![Alt text](https://github.com/Copticoder/CrashAgentPPO/blob/master/episode_vs_rewards_Crash.png "reward graph")
## Agent Performance

This video shows one episode from its highest reward return.


[![Crash Agent Performance](https://img.youtube.com/vi/5j5INyqFaUY/0.jpg)](https://www.youtube.com/watch?v=5j5INyqFaUY)


# Future Work
For Future work, there are some bottlenecks that might be addressed. For example, It might be that the CNN archetictures used were a bottle neck in this specific paradigm when dealing with retro environments. As the details of these environments are much more than the Atari environments with a DQN approach for example. A Vision Transformers (VIT) (Dosovitskiy et al.) Approach might be more effective. Moreover, Attention mechanisms for solving such problems may make the agent learn to process important parts of the frames like enemy locations, box locations in more details.
