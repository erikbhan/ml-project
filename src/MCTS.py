# ignore deprecation warnings ('safe' as long as we don't update packages)
from warnings import filterwarnings
filterwarnings("ignore")

from math import sqrt, log, inf
from copy import deepcopy

import gym
from gym_go.gogame import turn, random_weighted_action, random_action

import torch

import numpy as np

BOARD_SIZE = 5 # Board will be of size BOARD_SIZE * BOARD_SIZE (BOARD_SIZE**2)
ACTIONSPACE_LENGTH = BOARD_SIZE ** 2 + 1
UCB_C = 2
BLACK, WHITE, INVALID = 0, 1, 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Selection
#     - Taverse the tree to find greatest UCB-score
# 2. Expansion
#     - If the selected leaf node has been visited before expand by adding weighted game action
# 3. Rollout
#     - Simulate the game until end-condition from the expanded leaf
# 4. Back-propagation
#     - Updating the value of each ancestor node of the expanded leaf

class Node():
    def __init__(self, env, parent, action):
        self.env : gym.Env = env # This env will be altered by the other player
        self.value : int = 0 # Value estimate
        self.trials : int = 0 # Number of trials for this node
        self.parent : Node = parent # Parent node of this node
        self.children : list[Node] = [] # List of children of this node
        self.action : int = action # The step action made by this node
    
    # calculate a Upper Confidence Bound
    def ucb(self, total_trials):
        return self.value + ( UCB_C * sqrt(log(total_trials) / self.trials) )
    
    # Add a new node to a leaf node
    def expansion(self):
        for action in range(ACTIONSPACE_LENGTH - 1):
            x, y = action // BOARD_SIZE, action % BOARD_SIZE
            if self.env.state()[INVALID, x, y] == 0:
                child_env = deepcopy(self.env)
                child_env.step(action)
                self.children.append(Node(child_env, self, action))

        child_env = deepcopy(self.env)
        child_env.step(ACTIONSPACE_LENGTH - 1)       
        self.children.append(Node(child_env, self, ACTIONSPACE_LENGTH - 1))

    # Simulate game from current move until end-condition returning the score
    def rollout(self, move_selection_method):
        if self.env.done:
            return self.env.reward()
        
        rollout_env = deepcopy(self.env)
        done = False
        while not done:
            random_action = move_selection_method(rollout_env)
            _, _, done, _ = rollout_env.step(random_action)
        return rollout_env.reward()

    def __str__(self, level=0):
        ret = "    "*level
        if self.action:
            ret += f"{self.action:02d}: "
        else:
            ret += f"00: "
        ret += repr(self.value) + "\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

class Monte_Carlo_Tree_Search():
    def __init__(self, size, ml_model):
        self.env : gym.Env = gym.make('gym_go:go-v0', size=size, reward_method='heuristic')
        self.env.reset()
        self.root = Node(self.env, None, None)
        self.ml_model = ml_model
        if ml_model is not None:
            ml_model.to(device)
    
    # Gets the weights of all moves from the Machine Learning model
    def __get_move_weights(self, env : gym.Env):
        state = env.state()
        if self.ml_model is not None:
            move_weights = self.ml_model.forward(torch.tensor(state, device=device)).detach().cpu().numpy()
        else:
            move_weights = np.ones(ACTIONSPACE_LENGTH)
            for move in move_weights:
                move = move/len(move_weights)

        board_shape = state.shape[1:]
        for i in range(len(move_weights) - 1):
            action2d = i // board_shape[0], i % board_shape[1], i
            if state[INVALID, action2d[0], action2d[1]] == 1:
                move_weights[i] = 0.0
        return move_weights

    # Gets a weighted move for the given env
    def get_weighted_move(self, env : gym.Env):
        move_weights = self.__get_move_weights(env)
        return random_weighted_action(move_weights)

    # Update scores of all parent nodes after rollout
    def __back_propagation(self, rollout_node: Node, rollout_result):
        current_node = rollout_node
        while current_node != None:
            current_node.trials += 1
            # if turn(self.env.state()) == BLACK:
            #     current_node.value -= rollout_result
            # if turn(self.env.state()) == WHITE:                
            current_node.value += rollout_result
            current_node = current_node.parent
    
    # Find and return the leaf node with the highest UCB-score 
    def __selection(self, node: Node = None):
        if node is None:
            node = self.root
        selected_child = node
        current_node = node
        while len(current_node.children) > 0:
            current_best_ucb = -inf
            for child in current_node.children:
                if child.trials == 0:
                    return child

                child_ucb = child.ucb(self.root.trials)

                if child_ucb > current_best_ucb:
                    selected_child = child
                    current_best_ucb = child_ucb

            current_node = selected_child

        return selected_child
    
    # Explores the tree for the given number of iterations
    def run(self, iterations, node: Node = None):
        if node is None:
            node = self.root

        selected_node = node
        run = 0
        while run < iterations:
            selected_node = self.__selection(node)

            if not selected_node.env.done and selected_node.trials > 0:
                selected_node.expansion()
                selected_node = selected_node.children[0]

            rollout_result = selected_node.rollout(self.get_weighted_move)
            
            self.__back_propagation(selected_node, rollout_result)
            run += 1

    # Explores plays one game
    def run_game(self):
        # Run MCTS until a game is completed
        selected_node = self.root
        run = 1
        while not selected_node.env.done:
            # print("run:", run)
            selected_node = self.__selection()

            if selected_node.trials > 0:
                # print("Expanding node")
                selected_node.expansion()
                selected_node = selected_node.children[0]

            # print("Rollout")
            rollout_result = selected_node.rollout(self.get_weighted_move)
            self.__back_propagation(selected_node, rollout_result)
            # print("Done:", selected_node.env.done)
            run += 1
        # returns the node to allow for printing the game
        return selected_node
    
    # searches the tree for a spesific state
    def __find_node_from_state(self, state, node: Node = None):
        if node is None:
            node = self.root
        if np.array_equal(node.env.state(), state):
            return node

        for child in node.children:
            if np.array_equal(child.env.state(), state):
                return child
            
            res = self.__find_node_from_state(state, child)
            if res != None and np.array_equal(res.env.state(), state):
                return res

    # Attempts to find the best move from the tree by searching for the state and finding the best child for that state
    def get_move_from_env(self, env):
        if len(self.root.children) == 0:
            self.root.expansion()

        node = self.__find_node_from_state(env.state())
        self.run(15, node)

        best_child = None
        current_best_value = -inf
        for child in node.children:
            child_value = child.rollout(self.get_weighted_move)
            if child.value > current_best_value:
                best_child = child
                current_best_value = child_value

        if best_child != None:
            if len(best_child.children) == 0 and not best_child.env.done:
                best_child.expansion()
        return best_child

    # Makes a list of all states and a list of all move_weights for all expanded nodes in the tree
    def get_tree_data(self):
        x, y = [], []
        self.__get_node_data(self.root, x, y)
        return x, y

    # recurrsive tree traversal method for get_tree_data
    def __get_node_data(self, node, x, y):
        x.append(node.env.state())
        y.append([0] * (BOARD_SIZE**2 + 1))
        for child in node.children:
            y[-1][child.action] = child.value
            if len(child.children) > 0:
                self.__get_node_data(child, x, y)