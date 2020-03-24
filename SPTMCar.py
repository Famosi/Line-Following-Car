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

MIN = 100000000
STEP = 70


class SPTMMountainCar(object):

    def __init__(self, folder):
        self.graphs = {}  # saved trajectories
        self.line = []
        for episode_id, filename in enumerate(os.listdir(folder)):
            with open(folder + filename, "rb") as pickle_file:
                df = pickle.load(pickle_file)
                self.line = [tuple(r) for r in df.to_numpy().tolist()]

    def predict_rollout_head(self, n, denv):
        rotation = denv.rotation

        # We want to return a list of observations which represents the predicted rollout head
        def copy_env(environment):
            copy_env = gym.make("MountainCar-v0")
            copy_env.reset()
            copy_env.env.state = environment.env.state
            # copy_env.env.rotation = environment.env.rotation
            # Number of steps done is NOT saved
            return copy_env

        m = 0
        for i in range(0, n):
            m += 5 ** i

        nodes = list(range(1, m + 1))

        current_observation = denv.state
        current_position = current_observation[0]
        current_velocity = current_observation[1]
        done = bool(np.allclose(current_position, denv.goal_position, 0.1))

        current_parent = (nodes[0], denv)
        next_parents = []
        rollout = nx.DiGraph()

        rollout.add_node(nodes[0], position=denv.state[0], velocity=denv.state[1], action_sequence=[], node_sequence=[nodes[0]])
        nodes.pop(0)

        def __helper__(nodes, current_parent, next_parents, rollout, denv):
            if nodes:
                for action in [0, 1, 2, 3, 4]:
                    denv_ = copy_env(denv)
                    denv_.env.rotation = rotation
                    denv_.step(action)

                    dream_position = denv_.state[0]
                    dream_velocity = denv_.state[1]

                    done = bool(np.allclose(dream_position, denv.goal_position, 0.1))

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

                if not done:
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

        for _ in range(STEP):
            rollout = self.predict_rollout_head(4, dream_env)
            # tree_x = []
            # tree_y = []
            min_dist = MIN
            min_pair = (MIN, MIN)
            min_shift = MIN
            closest_point = []
            best_node = -1
            for node in rollout.nodes:
                # if it's not a leaf and it's not the root
                if rollout.succ[node] and node > 1:
                    position = rollout.nodes[node]['position']

                    # LOSS-1: get the closest point for this node
                    for point in self.line:
                        dist = np.linalg.norm(position - point[:2])
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = point

                    # LOSS-2: get the (more similar) shift
                    for k, v in rollout.succ[node].items():
                        k_pos = rollout.nodes[k]['position']
                        shift_point = closest_point[2:]
                        shift_node = np.array([position[0] - k_pos[0], position[1] - k_pos[1]])
                        dist_shift = np.linalg.norm(shift_node - shift_point)
                        if dist_shift < min_shift:
                            min_shift = dist_shift

                    # the node that has the min pair is the best node
                    pair = (min_shift, min_dist)
                    if pair < min_pair:
                        min_pair = pair
                        best_node = node

            # best_node_pos = rollout.nodes[best_node]['position']
            # line_to_follow_x = [x[0] for x in self.line]
            # line_to_follow_y = [y[1] for y in self.line]
            # plt.scatter(dream_env.state[0][0], dream_env.state[0][1], color='green')
            # plt.scatter(tree_x, tree_y, color='blue')
            # plt.scatter(best_node_pos[0], best_node_pos[1], color='red')
            # plt.plot(line_to_follow_x, line_to_follow_y, color='black')
            # plt.show()

            action_seq = rollout.nodes[best_node]['action_sequence']
            next_action = action_seq[0]

            dream_env.step(next_action)
            dream_env.render()

        dream_env.close()


if __name__ == '__main__':
    mc = SPTMMountainCar("./episodes_keyboard/")
    env = gym.make("MountainCar-v0")
    env.reset()
    dream_env = gym.make("MountainCar-v0")
    dream_env.reset()

    # Setting starting position (debugging)
    # dream_env.env.state[0] = np.array([0.10003285, 0.92225642])
    # print("starting position: ", dream_env.state[0])

    mc.dream_forward(dream_env)
