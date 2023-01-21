import numpy as np
import retro
import time 
from utils import wrapframe, CrashDiscretizer
from Agent import Agent
import cv2
from gym.wrappers import GrayScaleObservation,FrameStack,TransformObservation
def main():
    episode_num=500_000 #number of episodes
    rollout_horizon=20 # number of steps in each episode 
    num_actions=8 #number of actions crash can take
    batch_size=32
    num_epochs=5
    total_steps= episode_num*(rollout_horizon/batch_size)*num_epochs #The total number of gradient steps  
    env = retro.make(game='CrashBandicootTheHugeAdventure-GbAdvance',state="JungleJam")

    env=CrashDiscretizer(env) #This removes unnecessary button actions
    #converting the observations to grey scale, framestacking them and resizing them to (84,84), using the number of stacking frames to be 6
    env=GrayScaleObservation(env)  
    env=FrameStack(env,6)
    env = TransformObservation(env, wrapframe)   
    PPOAgent=Agent(num_actions)    
    for episode in range(episode_num):
        data=[]
        # env = Monitor(env, f"./videos/Crash_{episode}", force=True)
        movie_path=f"./videos/Crash-{episode}.bk2"
        observation = env.reset()
        for _ in range(rollout_horizon):
            action, log_prob, entropy, value = PPOAgent.predict_actor_critic(observation)
            observation, reward, done, _ = env.step(env.action_space.sample())
            # env.render()
            PPOAgent.store_rollout(observation,action,log_prob,entropy,value,np.round(reward,2),done)
            if done:
                observation = env.reset()
                break
        PPOAgent.learn()
        break
        # PPOAgent.learn()
        if episode%2==0:
            env.record_movie(movie_path)
            
if __name__ == '__main__':
    main()