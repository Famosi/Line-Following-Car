import gym
import pandas as pd
import numpy as np
import pickle

import gym

from baselines.baselines import deepq
from baselines.baselines.common import models


def main():
    """
    Collecting observations in the mountaincar environment by acting in an action space generated from a
    trained model. The episodes are stored in a DataFrame
    """
    env = gym.make("MountainCar-v0")

    ep_list = []
    obs_list = []
    episode = 1

    act = deepq.learn(
        env,
        network=models.mlp(num_layers=1, num_hidden=64),
        total_timesteps=0,
        load_path='./models/mountaincar_model_deepq.pkl'
    )

    for _ in range(1000):
        observation, done = env.reset(), False
        while not done:
            env.render()
            # Here dream several trajectories into the future using recursion. Check if it is possible from the
            # current position to switch to a position which lies on a better path.
            # Prior: At the beginning we don't have any information about the direction because the velocity is zero
            # So: Do a random step into a random direction or from an optimal policy; it does not matter
            # Remember the previous position and velocity

            observation, reward, done, info = env.step(act(observation[None])[0])
            obs_act = np.append(observation, act(observation[None])[0])
            ep_list.append(obs_act)
        # Save observations in a dataframe. Episodic.
        else:
            #ep = pd.DataFrame(ep_list, columns=["position", "velocity", "action"])
            ep = pd.DataFrame(ep_list, columns=["position_x", "position_y", "velocity_x", "velocity_y", "action"])
            ep["time"] = range(0, len(ep_list))
            print(ep)
            ep.to_pickle("./episodes_deepq_simple/episode_" + str(episode) + ".pkl")
            obs_list.append(ep)
            print(ep)
            ep_list = []
            episode += 1
            observation = env.reset()


if __name__ == '__main__':
    main()
