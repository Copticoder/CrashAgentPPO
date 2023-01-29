import numpy as np
import retro
import torch
from utils import wrapframe, CrashDiscretizer,save_scores
from Agent import Agent
from gym.wrappers import GrayScaleObservation,FrameStack,TransformObservation
import cv2 
from utils import *
episode_num=5000 #number of episodes
rollout_horizon=2048 #number of steps in each episode 
num_actions=8 #number of actions crash can take
batch_size=64
num_epochs=10
learning_rate=0.0003
total_steps= np.float32(episode_num*(rollout_horizon/batch_size)*num_epochs) #The total number of gradient steps  
env = retro.make(game='CrashBandicootTheHugeAdventure-GbAdvance',state="JustInSlimechk2")
#This removes unnecessary button actions
env=CrashDiscretizer(env) 
num_outputs=8
#converting the observations to grey scale, framestacking them and resizing them to (84,84), and then stacking frames
env = StochasticFrameSkip(env,4,0.3) 
# env = Downsample(env,2)
env= GrayScaleObservation(env)
#crop the frame from unuseful information
env=TransformObservation(env,lambda obs: obs[20:120, 90:200])  
env= FrameStack(env,4)
PPOAgent=Agent(num_actions=num_outputs,learning_rate=learning_rate,rollout_horizon=rollout_horizon,batch_size=batch_size,n_epochs=num_epochs)
# PPOAgent.restore_models(50)
score_history = []
n_steps = 0
steps=0

max_score=0
score = 0
print("Beginning Training....")
for episode in range(51,episode_num+1):
    data=[]
    movie_path=f"./videos/Crash-{episode}.bk2"
    observation = env.reset()
    #start recording this specific episode and save the models
    env.record_movie(movie_path)
    for _ in range(rollout_horizon):
        steps+=1
        observation = torch.FloatTensor(np.array(observation)).to(PPOAgent.device)
        dist, value = PPOAgent.model(observation)  # nn evaluate the observation
        action= dist.sample().cuda() if PPOAgent.use_cuda else dist.sample()
        next_observation, reward, done, _ = env.step(action.cpu().numpy()[0])
        reward=[np.around(reward,2)]
        done=[done]
        score+=reward[0]
        log_prob = dist.log_prob(action) 
        log_prob_vect = log_prob.reshape(len(log_prob), 1)
        action_vect = action.reshape(len(action), 1)
        PPOAgent.store_rollout(observation,action_vect,log_prob_vect,value,torch.FloatTensor(reward).unsqueeze(1).to(PPOAgent.device),torch.FloatTensor(done).unsqueeze(1).to(PPOAgent.device))
        observation = next_observation
        env.render()

    next_observation = torch.FloatTensor(np.array(next_observation)).to(PPOAgent.device) # consider last observation of the collection step for the bootstraping in the GAE step
    next_observation = torch.div(next_observation,255)

    _, next_value = PPOAgent.model(next_observation) # collect last value effect of the last collection step
    PPOAgent.values += [next_value]
    PPOAgent.values.append(next_value)

    PPOAgent.learn()  
    # save the scores of the whole training episodes 
    score_history.append(score)
    avg_score=np.average(score_history[-100:])
    print('episode',episode,'score %.1f' % score, 'avg score %.1f' % avg_score,
        'time_steps', steps)
    if episode%50==0 or max_score<=score:
        #close recording this episode 
        PPOAgent.save_models(episode)
        max_score=score
        env.stop_record()
        save_scores(scores=score_history)
    score=0


    
    
