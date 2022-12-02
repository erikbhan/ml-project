import numpy as np
import gym

from math import sqrt, log, inf
from copy import deepcopy

BOARD_SIZE = 5
UCB_CONSTANT = 2

class Node():

    def __init__(self, parent=None, env=None, move=None):
        self.parent, self.env, self.move = parent, env, move
        if env == None:
            self.env = gym.make('gym_go:go-v0', size=BOARD_SIZE, komi=0, reward_method="heuristic")
            self.env.reset()
        self.children, self.value, self.visits = [], 0, 0

    def __str__(self, level=0):
        ret = "    "*level
        if self.move:
            ret += f"{self.move:02d}: "
        else:
            ret += f"00: "
        ret += repr(self.value) + "\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

    def __ucb(self, total_visits):
        if self.visits == 0: return inf
        return self.value + (UCB_CONSTANT * sqrt(log(total_visits) / self.visits))

    def __selection(self):
        if len(self.children) == 0: return self
        total_visits = self.__total_visits()
        best_child = self.children[0]
        best_child_ucb = best_child.__ucb(total_visits)
        for child in self.children:
            if child.env.done: continue # in case the best child is finished, pick another
            if child.__ucb(total_visits) > best_child_ucb: best_child = child
        return best_child.__selection()

    def __expansion(self):
        if self.env.done: return None
        for move in np.argwhere(self.env.valid_moves()).flatten():
            temp_env = deepcopy(self.env)
            temp_env.step(move)
            self.children.append(Node(self, temp_env, move))
        return self.children[0]

    def __backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent != None: self.parent.__backpropagate(value)

    def __rollout(self):
        temp_env = deepcopy(self.env)
        while not temp_env.done:
            temp_env.step(temp_env.uniform_random_action())
        self.__backpropagate(temp_env.reward())

    def mcts(self, iterations):
        for _ in range(iterations):
            node = self.__selection()
            if node.visits != 0: node = node.__expansion()
            if node == None: 
                continue
            node.__rollout()

    def __total_visits(self):
        if self.parent == None: return self.visits
        return self.parent.__total_visits()

    # Basically the same as selection, maybe we can find a way to integrate them
    def best_move(self, white=False):
        if len(self.children) == 0: return self
        total_visits = self.__total_visits()
        best_child = self.children[0]
        best_child_ucb = best_child.__ucb(total_visits)
        for child in self.children:
            if white:
                if child.__ucb(total_visits) < best_child_ucb:
                    best_child = child
            else:
                if child.__ucb(total_visits) > best_child_ucb:
                    best_child = child
        return best_child.move()