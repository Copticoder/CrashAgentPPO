import numpy as np
import retro
import torch
from utils import wrapframe, CrashDiscretizer,save_scores
from Agent import Agent
from gym.wrappers import GrayScaleObservation,FrameStack,TransformObservation

episode_num=1000 #number of episodes
rollout_horizon=2048 #number of steps in each episode 
num_actions=8 #number of actions crash can take
batch_size=64
num_epochs=10
learning_rate=0.005
total_steps= np.float32(episode_num*(rollout_horizon/batch_size)*num_epochs) #The total number of gradient steps  
env = retro.make(game='CrashBandicootTheHugeAdventure-GbAdvance',state="JustInSlime")
#This removes unnecessary button actions
env=CrashDiscretizer(env) 
num_outputs=8
#converting the observations to grey scale, framestacking them and resizing them to (84,84), and then stacking frames
env=GrayScaleObservation(env)  
env=FrameStack(env,4)
env = TransformObservation(env, wrapframe)   
PPOAgent=Agent(num_actions=num_outputs,learning_rate=learning_rate,rollout_horizon=rollout_horizon,batch_size=batch_size,n_epochs=num_epochs)
# PPOAgent.restore_models(50)
score_history = []
n_steps = 0
observation = env.reset()
steps=0

print("Beginning Training....")
for episode in range(1,episode_num+1):
    score = 0
    data=[]
    movie_path=f"./videos/Crash-{episode}.bk2"
    observation = env.reset()
    if episode in [1,50,100,250,500,750,1000,1250,1500]:
        #start recording this specific episode and save the models
        env.record_movie(movie_path)
        PPOAgent.save_models(episode)
    for _ in range(rollout_horizon):
        steps+=1
        observation = torch.FloatTensor(observation).to(PPOAgent.device)
        dist, value = PPOAgent.model(observation)  # nn evaluate the observation
        action= dist.sample().cuda() if PPOAgent.use_cuda else dist.sample()
        next_observation, reward, done, _ = env.step(action.cpu().numpy()[0])
        reward=[reward]
        done=[done]
        score+=reward[0]
        log_prob = dist.log_prob(action) 
        log_prob_vect = log_prob.reshape(len(log_prob), 1)
        action_vect = action.reshape(len(action), 1)
        PPOAgent.store_rollout(observation,action_vect,log_prob_vect,value,torch.FloatTensor(reward).unsqueeze(1).to(PPOAgent.device),torch.FloatTensor(done).unsqueeze(1).to(PPOAgent.device))
        observation = next_observation
        env.render()
    next_observation = torch.FloatTensor(next_observation).to(PPOAgent.device) # consider last observation of the collection step for the bootstraping in the GAE step
    _, next_value = PPOAgent.model(next_observation) # collect last value effect of the last collection step
    PPOAgent.values += [next_value]
    PPOAgent.values.append(next_value)

    if episode in [1,50,100,250,500,750,1000,1250,1500]:
        #close recording this episode 
        env.stop_record()

    PPOAgent.learn()  
# save the scores of the whole training episodes 
    score_history.append(score)
    avg_score=np.average(score_history[-100:])
    print('score %.1f' % score, 'avg score %.1f' % avg_score,
        'time_steps', steps)
save_scores(scores=score_history)


    
    
