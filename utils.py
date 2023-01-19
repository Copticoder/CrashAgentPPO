import cv2
import numpy as np
import gym


def wrapframe(frame):
    """Converts the frames into grey scale and resizes them into 84x84 images"""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(
        frame, (84, 84), interpolation=cv2.INTER_AREA
    )
    return frame


def stackframes(data, k=4):
    """Frame stacking function which takes a multiple steps in the 
    environment and stacks them on top of each other for the temporal
    information for the agent.
    """
    rollout = []
    rewards= []
    for stack in range(0, len(data)-k, k):
        framestack = []
        rewardstack = []

        for step in range(min(k, len(data)-stack)):
            framestack.append(wrapframe(data[stack+step][0]))
            rewardstack.append(data[stack+step][1])
        if len(framestack) < k:
            framestack.append(framestack[-1]*(k-len(framestack)))

        # calculate the average of the rewards in the stacked steps and round them
        reward = sum(rewardstack)/k
        reward = np.round(reward, 2)
        rewards.append(reward)
        framestack = np.array(framestack)
        # transpose the (k,84,84) shape to (84,84,k)
        framestack = np.transpose(framestack, (1, 2, 0))
        rollout.append(framestack)
    return np.array(rollout,dtype=np.float32),np.array(rewards,dtype=np.float32)


"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
"""


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(
            len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class CrashDiscretizer(Discretizer):
    """
    Use Crash-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=[['LEFT'], ['RIGHT'], [
            'LEFT', 'R'], ['RIGHT', 'R'], ['A'], ['R'], ['B'], [None]])
