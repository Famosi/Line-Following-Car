import pandas as pd
import os
import pickle
import numpy as np

path = "./episodes/"


def main():
    list_ = []
    for filename in os.listdir(path):
        with open(path + filename, 'rb') as pickle_file:
            current_episode = pickle.load(pickle_file)
            list_.extend(np.array(current_episode[["position", "velocity", "action"]].values))

    all_eps = pd.DataFrame(np.array(list_), columns=["position", "velocity", "action"])
    all_eps.to_pickle(path + "all_episodes.pkl")


if __name__ == '__main__':
    main()
