import numpy as np
import retro
import time 
from utils import wrapframe, CrashDiscretizer
from Agent import Agent
import cv2
from gym.wrappers import GrayScaleObservation,FrameStack,TransformObservation
def main():
    episode_num=500_000 #number of episodes
    rollout_horizon=1024 # number of steps in each episode 
    num_actions=8 #number of actions crash can take
    batch_size=16
    num_epochs=5
    total_steps= np.float32(episode_num*(rollout_horizon/batch_size)*num_epochs) #The total number of gradient steps  
    env = retro.make(game='CrashBandicootTheHugeAdventure-GbAdvance',state="JungleJam")

    env=CrashDiscretizer(env) #This removes unnecessary button actions
    #converting the observations to grey scale, framestacking them and resizing them to (84,84), using the number of stacking frames to be 6
    env=GrayScaleObservation(env)  
    env=FrameStack(env,6)
    env = TransformObservation(env, wrapframe)   
    PPOAgent=Agent(num_actions=num_actions,rollout_horizon=rollout_horizon,total_steps=total_steps,batch_size=batch_size,n_epochs=num_epochs)    

    score_history = []
    n_steps = 0
    print("Beginning Training....")
    for episode in range(1,episode_num+1):
        score = 0
        data=[]
        movie_path=f"./videos/Crash-{episode}.bk2"
        observation = env.reset()
        
        if episode in [1,50,100,250,500,1000,2500,5000,10_000,15_000,20_000,25_000,50_000,100_000,200_000]:
            #start recording this specific episode and save the models
            env.record_movie(movie_path)
            PPOAgent.save_models(episode)
        
        for _ in range(rollout_horizon):
            action, log_prob, value = PPOAgent.predict_actor_critic(observation)
            observation, reward, done, _ = env.step(action[0])
            score +=reward
            n_steps+=1
            # time.sleep(0.05)
            # env.render()
            PPOAgent.store_rollout(observation,action,log_prob,value,np.round(reward,2),done)
            if done:
                observation = env.reset()

        if episode in [1,50,100,250,500,1000,2500,5000,10_000,15_000,20_000,25_000,50_000,100_000,200_000]:
            #close recording this episode 
            env.stop_record()

        score_history.append(score)
        avg_score=np.average(score_history[-100:])
        print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', episode*(rollout_horizon/batch_size)*num_epochs)
        PPOAgent.learn()  
        
if __name__ == '__main__':
    main()