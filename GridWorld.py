import itertools
import math
import numpy.random as rand
import numpy as np
from mdptoolbox.mdp import ValueIteration

class GridWorld(object):

    def __init__(self, n=3, m=4):
        self.rows = n
        self.cols = m
        self.states = set(itertools.product(range(n), range(m)))
        self.blocks = set([(1,1)])
        self.end_states = set([(0, 3), (1, 3)])
        self.reward_dict = self.set_starting_reward()
        self.actions = set(['up', 'down', 'left', 'right'])
        self.utilities = self.get_utility()
        self.policy = {}

    def set_starting_reward(self):
        d = {}
        for state in self.states:
            if state not in self.blocks:
                d[state] = -.03
        
        d[(0,3)] = 1
        d[(1, 3)] = -1
        return d

    def get_neighbors(self, state):

        def invalid(s):
            r, c = s
            return not(r < 0 or r >= self.rows or c < 0 or c >= self.cols or s in self.blocks)
        r, c = state
        candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        candidates = list(filter(invalid, candidates))
        return candidates

    def get_next_state(self, s, action):
        r, c = s
        candidates = self.get_neighbors(s)
        s_prime = {
            'up': (r-1, c),
            'right': (r, c+1),
            'down': (r+1, c),
            'left': (r, c-1)
        }[action]
        return s_prime if s_prime in candidates else s

    def get_prob_actions(self, action):
        return [action, 'left', 'right'] if action in ['up', 'down'] else [action, 'up', 'down']

    def value_iteration(self, state, gamma):
        prev_utility = {state: 0}
        probs = [.8, .1, .1]
        curr_state = state
        while not (prev_utility == self.utilities):
            # print(np.mean(list(prev_utility.values())), np.mean(list(self.utilities.values())))
            curr_utilities = self.utilities.copy()
            for s in self.states - self.blocks:
                if s in self.end_states:
                    continue

                best_action = None
                max_score = -math.inf
                for action in self.actions:
                    s_primes = [self.get_next_state(s, a) for a in self.get_prob_actions(action)]
                    # print(s_primes)
                    score = sum([self.utilities[s_primes[i]] * probs[i] for i in range(len(probs))])
                    if score > max_score:
                        max_score = score
                        best_action = action
                
                self.policy[s] = best_action
                curr_utilities[s] = self.reward_dict[s] + gamma * max_score
            prev_utility = self.utilities
            self.utilities = curr_utilities

                
    def get_max_score(self, state):
        probs = [.6, .2, .2]
        for action in self.actions:
            outcomes = self.get_prob_actions(action)
            s_primes = [self.get_next_state(state, a) for a in outcomes]


    def get_utility(self):
        d = {}
        for s in self.states:
            d[s] = 0
        d[(0, 3)] = 1
        d[(1, 3)] = -1
        return d

    def print_policy(self):
        output = [['' for i in range(self.cols)] for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in self.end_states.union(self.blocks):
                    continue
                output[i][j] = {
                    'up': '^',
                    'down': 'v',
                    'left': '<-',
                    'right': '->'
                }[self.policy[(i, j)]]
        for p in output:
            print(p)
def main():
    gw = GridWorld()
    gw.value_iteration((0, 2), .9)
    # print(gw.utilities)
    gw.print_policy()

if __name__ == "__main__":
    main()
        
    