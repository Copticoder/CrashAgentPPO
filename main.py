import numpy as np
import retro
import time 
from utils import stackframes, CrashDiscretizer
from Agent import Agent
def main():
    env = retro.make(game='CrashBandicootTheHugeAdventure-GbAdvance',state="JungleJam")
    env=CrashDiscretizer(env)
    for episode in range(5):
        data=[]
        
        movie_path=f"./videos/Crash-{episode}.bk2"
        observation = env.reset()
        for _ in range(1025):
            observation, reward, done, info = env.step(env.action_space.sample())
            data.append([observation,reward,done,info])
            if done:
                obs = env.reset()
                break
        rollout, rewards = stackframes(data,4)
        print(rollout.shape)
        testAgent=Agent(rollout,8)
        raise

        if episode%2==0:
            env.record_movie(movie_path)
            
if __name__ == '__main__':
    main()