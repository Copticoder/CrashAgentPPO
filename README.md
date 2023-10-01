# Abstract on CrashAgentPPO
This repo contains my implementation for a variant of the PPO reinforcement learning algorithm on the retro environments by openAI.
Specifically, on Crash Bandicoot The Huge Adventure game. Moreover, The results of the learning experiment were noisy resembling most of reinforcement learning experiments. This implementation and training was done on my personal laptop on a mid-range GPU Nvidia 1650 4GB memory. This project is for educational and demonstrative purposes.
# KeyFiles 
* Agent.py : Contans the agent class, including the agent memory, neural networks initialization, the learning algorithm (PPO), training loop and finally saving and loading the models
* networks.py : contains the classes for the backbone (Shared CNN network), the actor head, the critic head and finally the whole ActorCritic archeticture.
* main.py : The program that runs the crash environment and begins rolling out episodes and collecting data for the agnet as well as saving episode recordings and score results.
* reward.lua : contains the program for reshaping the rewards coming from the openAI gameboy advance environments. Furthermore, editing the score coming like lives remaining, number of apples eaten, number of boxes broken and so on..
* utils.py : a file that contains some helper functions and openAI wrappers for the environment.
# Methodolgy
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
# Citations

- OpenAI Gym:
  Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym. arXiv preprint arXiv:1606.01540.

- Retro:
  Nichol, A., Pfau, V., Hesse, C., Klimov, O., & Schulman, J. (2018). Gotta Learn Fast: A New Benchmark for Generalization in RL. arXiv preprint arXiv:1804.03720.

- Proximal Policy Optimization (PPO):
  Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

- PyTorch:
  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc.

- Crash Bandicoot The Huge Adventure:
  Vivendi Universal Games. (2002). Crash Bandicoot The Huge Adventure.

- Vision Transformers (VIT):
  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
  
- ActorCritic:
  Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International conference on machine learning   (pp. 1928-1937).

- Convolutional Neural Networks: 
  Convolutional Neural Networks (CNNs): LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. nature, 521(7553), 436-444.
  
- Deep Q-Networks:
  Volodymyr Mnih, Kavukcuoglu Koray, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533,     February 2015.
