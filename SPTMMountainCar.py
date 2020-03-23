import pickle
import os
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import networkx as nx
from itertools import groupby
from collections import Counter
from pynput import keyboard
import gym
import time

MIN_DIST = 100000000


class SPTMMountainCar(object):

    def __init__(self, folder):
        self.graphs = {}  # saved trajectories
        self.line = []
        for episode_id, filename in enumerate(os.listdir(folder)):
            with open(folder + filename, "rb") as pickle_file:
                df = pickle.load(pickle_file)
                self.line = [tuple(r) for r in df.to_numpy().tolist()]


    def predict_rollout_head(self, n, denv):

        # We want to return a list of observations which represents the predicted rollout head

        def copy_env(environment):
            copy_env = gym.make("MountainCar-v0")
            copy_env.reset()
            copy_env.env.state = environment.env.state
            # Number of steps done is NOT saved
            return copy_env

        m = 0
        for i in range(0, n):
            m += 3 ** i

        nodes = list(range(1, m + 1))

        current_observation = denv.state
        current_position = current_observation[0]
        current_velocity = current_observation[1]
        done = bool(np.allclose(current_position, denv.goal_position, 0.05))

        current_parent = (nodes[0], denv)
        next_parents = []
        rollout = nx.DiGraph()

        rollout.add_node(nodes[0], position=denv.state[0], velocity=denv.state[1], action_sequence=[], node_sequence=[nodes[0]])
        nodes.pop(0)

        def __helper__(nodes, current_parent, next_parents, rollout, denv):

            if nodes:
                for action in [0, 1, 2]:
                    denv_ = copy_env(denv)
                    denv_.step(action)
                    dream_position = denv_.state[0]
                    dream_velocity = denv_.state[1]
                    done = bool(np.allclose(dream_position, denv.goal_position, 0.05))

                    if not done:
                        next_parents.append((nodes[0], denv_))

                    current_action_sequence = rollout.nodes[current_parent[0]]["action_sequence"].copy()
                    current_action_sequence.append(action)
                    current_node_sequence = rollout.nodes[current_parent[0]]["node_sequence"].copy()
                    current_node_sequence.append(nodes[0])
                    rollout.add_node(nodes[0], position=np.array(dream_position), velocity=np.array(dream_velocity), action_sequence=current_action_sequence,
                                     node_sequence=current_node_sequence)
                    rollout.add_edge(current_parent[0], nodes[0], action=action)
                    nodes.pop(0)

                current_parent = next_parents.pop(0)

                if current_parent:
                    return __helper__(nodes, current_parent, next_parents, rollout, current_parent[1])
                else:
                    return rollout

            else:
                return rollout

        if not done:
            return __helper__(nodes, current_parent, next_parents, rollout, denv)

        return rollout

    def dream_forward(self, dream_env):
        line_to_follow = self.line
        for _ in range(0, 20):
            min_dist = MIN_DIST
            best_node = -1
            rollout = self.predict_rollout_head(4, dream_env)
            for node in rollout.nodes:
                if not rollout[node]:
                    position = rollout.nodes[node]['position']
                    for point in line_to_follow:
                        dist = np.linalg.norm(position - point[:2])
                        if dist < min_dist:
                            min_dist = dist
                            best_node = node

            action_seq = rollout.nodes[best_node]['action_sequence']
            print("ActionSeq:", action_seq)
            next_action = action_seq[0]
            print("NextAction:", next_action)
            dream_env.step(next_action)
            dream_env.render()
        dream_env.close()


if __name__ == '__main__':
    mc = SPTMMountainCar("./episodes_keyboard/")
    env = gym.make("MountainCar-v0")
    env.reset()
    dream_env = gym.make("MountainCar-v0")
    dream_env.reset()

    # Setting some starting positions for debugging purpose+
    dream_env.env.state[0] = np.array([0.10003285, 0.92225642])

    print("starting position: ", dream_env.state[0])

    mc.dream_forward(dream_env)
