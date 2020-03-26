import gym
import pandas as pd
import numpy as np

import pickle

import gym
from pynput import keyboard


# from baselines import deepq
# from baselines.common import models

MAX_STEP = 70


class PlayMountainCar(object):
    """
    Collecting observations in the mountaincar environment by acting in an action space generated from a
    trained model. The episodes are stored in a DataFrame
    """

    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.done = False
        self.step = 0
        self.ep_list = []
        self.obs_list = []
        self.observation = []

    def on_press(self, key):
        try:
            k = key.char  # single-char keys

        except:
            k = key.name  # other keys
        if key == keyboard.Key.esc or self.done or self.step > MAX_STEP:
            self.done = True
            return False  # stop listener

        if k == 'left':
            #Push left
            print('Key pressed: ' + k)
            self.observation, _, self.done, _ = self.env.step(1)
            self.step += 1
            self.ep_list.append(self.observation[0])
            self.env.render()
        if k == 'up':
            # push straight
            print('Key pressed: ' + k)
            self.observation, _, self.done, _ = self.env.step(0)
            self.step += 1
            self.ep_list.append(self.observation[0])
            self.env.render()
        if k == 'right':
            # push right
            print('Key pressed: ' + k)
            self.observation, _, self.done, _ = self.env.step(2)
            self.step += 1
            self.ep_list.append(self.observation[0])
            self.env.render()

    def play_and_collect(self):
        self.observation = self.env.reset()
        print("Pre-recording Movements")

        for _ in range(1000):
            print("Lets replay it!")
            with keyboard.Listener(on_press=self.on_press) as listener:
                listener.join()
            if self.done:
                break
        self.env.close()

        ep = pd.DataFrame(self.ep_list, columns=["position_x", "position_y"])
        print(ep["position_x"].shape[0])
        print(ep["position_x"][0])
        ep.to_pickle("./episodes_keyboard/episode_" + str(np.round(ep["position_x"][0], decimals=3)) + "_" +
                     str(ep["position_x"].shape[0]) + ".pkl")
        self.obs_list.append(ep)
        print(ep)
        self.ep_list = []


if __name__ == '__main__':
    mc = PlayMountainCar()
    mc.play_and_collect()
