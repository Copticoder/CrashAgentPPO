import cv2
import numpy as np
import gym
import pickle


def wrapframe(obs):
    """Converts the frames into grey scale and resizes them into 84x84 images"""
    observations = np.array([cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in obs],dtype=np.float32)
    return observations

def save_scores(scores):
    file_name = "scores.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(scores, f)

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



