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


class SPTMMountainCar(object):

    def __init__(self, folder):
        self.graphs = {}  # saved trajectories
        self.action_space = np.array([[1., 0.3], [1., 1.], [0.3, 1.]])

        # Discarded clustering approach with KMeans
        # self.n_clusters = n_clusters
        """
        Loads observations from pickle files and saves them as vertices. Episodes are dataframes with
        columns = "Position", "Velocity" and "Action"
        The position, velocity and action are the observations from a mountaincar environment. Every episode is a
        successful path towards the goal. 
        The observations are taken from trajectorys collected via keyboard.
        :param folder: The folder in which the episodes are stored.
        """
        # with open("./all_episodes.pkl", 'rb') as data:
        #     episodes = pickle.load(data)

        # self.KM = KMeans(n_clusters=self.n_clusters).fit(episodes[["position", "velocity"]].values)

        for episode_id, filename in enumerate(os.listdir(folder)):
            with open(folder + filename, "rb") as pickle_file:
                episode = pickle.load(pickle_file)
                # episode_labels = []
                episode_graph = nx.DiGraph()
                # print(episode[["position", "velocity", "action"]].values)
                current_action_sequence = []
                for index, value in enumerate(episode[["position_x", "position_y", "velocity_x", "velocity_y", "action"]].values):
                    # print("#########DEEEEBUGGING#########")
                    # print("The index currently isss:")
                    # print(index)
                    # print("The first item of value isssss:")
                    # print(value[0])
                    # print("The second item of value issssss:")
                    # print(value[1])
                    current_action_sequence.append(int(value[4]))
                    episode_graph.add_node(index, position=value, velocity=value, action_sequence=current_action_sequence)
                    if index > 0:
                        # print("index -1 isss")
                        # print(index-1)
                        # print("index is")
                        # print(index)
                        # print("The action for the previous action done wass")
                        # print(int(episode["action"][index-1]))
                        episode_graph.add_edge(index - 1, index, action=int(episode["action"][index - 1]))

            self.graphs[episode_id] = episode_graph  # saved trajectory
            # for u, v, d in episode_graph.edges(data=True):
            # main_graph_dict = nx.get_edge_attributes(self.graph, "weight")

            # if (u, v) in main_graph_dict.keys():
            #     self.graph[u][v]["weight"] = self.graph[u][v]["weight"] + d["weight"]
            # else:
            # self.graph.add_edge(u, v, weight=d["weight"])
            # self.graph.add_edge(u, v)

            # print("Composed Graph")
            # print(self.graph.nodes)
            # print(self.graph.edges)
            # print(nx.get_edge_attributes(self.graph, 'weight'))

        # Goal: Composing all trajectories to a connected Directed Graph.
        # Problem: The observations of each time step from a successful path towards the goal
        # consists of a position (-1.2, 0.6)
        # and a velocity (-0.07, 0.07). They have continuous values.
        # When we want to compose 2 paths to a bigger main graph there wont be no identical observation pair
        # because the values are continuous and therefore always different.
        # What we must do first therefore is to discretize the observations.
        # Manually labeling is not allowed so we should choose algorithms which does that for us.
        # A solution would be to take one episode and reduce the feature space by
        # assigning each observation to a cluster where then we end up with succession of clusters
        # and not the complex continuous observation pair which represents position and velocity.
        # But here comes another problem: Any clustering algorithm will assign the cluster number differently.
        # We dont really know which cluster number belongs to which cluster and therefore we can't compose
        # them because for one observation tuple in path 1 with a high chance a different cluster label
        # is assigned than for the other observation tuple in path 2, even though they could match.

        # The graph will grow linearly with the added observations.
        # Having 100k observations makes it unfeasible to combine different paths to a meaningful graph

    # def visualize_graph(self):
    #     nx.draw(self.graph, with_labels=True)
    #     # print(self.graph.edges)
    #     plt.show()

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
                    rollout.add_node(nodes[0], position=dream_position, velocity=dream_velocity, action_sequence=current_action_sequence,
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
        trg_render = True

        rollout = self.predict_rollout_head(2, dream_env)

        def get_jump_candidates(r, g):
            jump_candidates = {}
            for r_node in r.nodes:
                # Compare it to the other nodes
                for graph_id in g.keys():  # saved trajectories
                    for graph_node in g[graph_id].nodes:
                        # current_pos_delta = g[graph_id].nodes[graph_node]["position"][0] - rollout.nodes[r_node]["position"][0]
                        # current_velo_delta = g[graph_id].nodes[graph_node]["velocity"][0] - rollout.nodes[r_node]["velocity"][0]
                        current_pose_graph = [g[graph_id].nodes[graph_node]["position"][0], g[graph_id].nodes[graph_node]["position"][1]]
                        current_pose_rollout = rollout.nodes[r_node]["position"]
                        # print(current_pose_graph, current_pose_rollout)
                        try:
                            if bool(np.allclose(current_pose_graph[0], current_pose_rollout[0], 0.05) and np.allclose(current_pose_graph[1], current_pose_rollout[1], 0.05)):
                                # print('current_pos_delta', current_pos_delta, 'current_velo_delta', current_velo_delta)
                                jump_candidates[(r_node, graph_id, graph_node)] = g[graph_id].number_of_nodes() - graph_node
                        except ValueError as e:
                            break

            return jump_candidates

        # Rollout could be empty when at the goal position. If not then do the following
        if rollout:
            jump_candidates = get_jump_candidates(rollout, self.graphs)

            # If there is a trajectory to hop on to
            if jump_candidates:
                min_candidate = min(jump_candidates, key=jump_candidates.get)  # sort the candidates by the distance to the reward

                best_trajectory = self.graphs[min_candidate[1]]  # Followed Trajectory
                current_trajectory = nx.DiGraph()
                current_trajectory.add_node(current_trajectory.number_of_nodes(), position=dream_env.env.state[0], velocity=dream_env.env.state[1],
                                            action_sequence=[])
                for action in rollout.nodes[min_candidate[0]]["action_sequence"]:  # Take actions from the dreamed rollout
                    current_action_sequence = current_trajectory.nodes[current_trajectory.number_of_nodes() - 1]["action_sequence"].copy()
                    dream_env.step(action)
                    if trg_render:
                        dream_env.render()
                    current_action_sequence.append(action)
                    current_trajectory.add_node(current_trajectory.number_of_nodes(), position=dream_env.env.state[0],
                                                velocity=dream_env.env.state[1],
                                                action_sequence=current_action_sequence)

                # Stepping towards the action sequence of the best trajectory (stepping over the bridge)
                current_action_sequence = current_trajectory.nodes[current_trajectory.number_of_nodes() - 1]["action_sequence"].copy()
                if min_candidate[2] != 0:
                    new_action = self.graphs[min_candidate[1]].edges[min_candidate[2] - 1, min_candidate[2]]["action"]

                # If a candidate pair is not the very first node of the followed trajectory means no edge exists towards the first node just
                # execute the action to the 2nd node
                else:
                    new_action = self.graphs[min_candidate[1]].edges[min_candidate[2], min_candidate[2] + 1]["action"]

                dream_env.step(new_action)


                if trg_render:
                    dream_env.render()
                current_action_sequence.append(new_action)
                current_trajectory.add_node(current_trajectory.number_of_nodes(), position=dream_env.env.state[0],
                                            velocity=dream_env.env.state[1],
                                            action_sequence=current_action_sequence)

                # Execute actions from the optimal trajectory from now on
                for action in best_trajectory.nodes[best_trajectory.number_of_nodes() - 1]["action_sequence"][min_candidate[2]:len(best_trajectory.nodes[best_trajectory.number_of_nodes() - 1]["action_sequence"])]:
                    current_action_sequence = current_trajectory.nodes[current_trajectory.number_of_nodes() - 1]["action_sequence"].copy()

                    _, _, done, _ = dream_env.step(action)

                    current_action_sequence.append(action)
                    current_trajectory.add_node(current_trajectory.number_of_nodes(), position=dream_env.env.state[0],
                                                velocity=dream_env.env.state[1], action_sequence=current_action_sequence)

                    if trg_render:
                        dream_env.render()

                if trg_render:
                    dream_env.close()
                return current_trajectory, best_trajectory, [rollout.nodes[min_candidate[0]]["position"],
                                                             self.graphs[min_candidate[1]].nodes[min_candidate[2]]["position"]], [
                           rollout.nodes[min_candidate[0]]["velocity"], self.graphs[min_candidate[1]].nodes[min_candidate[2]]["velocity"]]

            # Execute a random action if no suitable trajectory could be found and then search again for bridge points
            else:
                choice = np.random.randint(0, 3)
                dream_env.step(choice)
                return self.dream_forward(dream_env)
        else:
            # When the dreamed rollout is at the goal position
            return nx.DiGraph(), nx.DiGraph(), [], []


def print_tree(t):
    for node in list(t.nodes):
        if t[node]:
            print(node, "->", t[node])


if __name__ == '__main__':
    mc = SPTMMountainCar("./episodes_keyboard/")
    env = gym.make("MountainCar-v0")
    env.reset()
    dream_env = gym.make("MountainCar-v0")
    dream_env.reset()

    # Setting some starting positions for debugging purpose+
    dream_env.env.state[0] = np.array([0.47180236, 0.32295259])
    # dream_env.env.state[0] = -0.48145347
    # dream_env.env.state[0] = -0.53145347
    # dream_env.env.state[0] = -0.55145347
    # dream_env.env.state[0] = -0.58145347

    print("starting pos")
    print(dream_env.state)

    current_trajectory, best_trajectory, jump_coords_pos, jump_coords_velo = mc.dream_forward(dream_env)