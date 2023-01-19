import retro, gym
from gym.wrappers import RecordVideo

# Create an instance of the gym environment
env = retro.make(game='CrashBandicootTheHugeAdventure-GbAdvance',state="JungleJam")

# Create an instance of the VideoRecorder wrapper
# env = RecordVideo(env, video_folder='videos/', episode_trigger=lambda episode_id: episode_id % 10 == 0)

# Define the number of episodes
n_episodes = 20
# Run the agent in the environment for the specified number of episodes
for episode in range(n_episodes):
    movie_path=f"./videos/Crash-{episode}.bk2"
    # Reset the environment
    obs = env.reset()
    done = False
    for i in range(100):
        # Perform an action
        action = env.action_space.sample() # this is an example of random action selection
        obs, reward, done, info = env.step(action)
    if episode%2==0:
        env.record_movie(movie_path)
    # Close the environment at the end of the episode
    # env.close()