import numpy as np
import collections

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from rlcard.utils.utils import *

class CFRModel:
    """
    Implementation of Monte Carlos CFR algorithm
    """

    def __init__(self, env, model_path):
        """
        :param env: game environment
        :param model_path: path to save/load model
        """

        self.use_raw = False

        self.env = env
        self.model_path = model_path

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0

        self.load()

    def train(self):
        """
        Do a single iteration of training
        """

        self.iteration += 1
        # Use multithreading for parallelizing traversal across players
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.parallel_traverse_tree, player_id)
                for player_id in range(self.env.num_players)
            ]
            for future in futures:
                future.result()  # Ensure all threads complete execution

        # Update policy after traversing
        self.update_policy()

    def parallel_traverse_tree(self, player_id):
        """
        Reset environment and start tree traversal for a specific player.
        """

        self.env.reset()
        probs = np.ones(self.env.num_players)
        self.traverse_tree(probs, player_id)

    def traverse_tree(self, probs, player_id):
        """
        Traverse the game tree
        :param probs: reach probability of current node for players
        :param player_id: player to update value
        :return: list of expected utilities for all players
        """

        if self.env.is_over():
            try:
                return self.env.get_payoffs()
            except Exception:
                return np.zeros(self.env.num_players)

        current_player = self.env.get_player_id()

        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        # If it is current player, we record the policy and compute regret
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                                np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        for action in legal_actions:
            action_prob = action_probs[action]
            regret = counterfactual_prob * (action_utilities[action][current_player]
                    - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * player_prob * action_prob
        return state_utility

    def update_policy(self):
        """
        Update policy based on current regrets
        """

        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def regret_matching(self, obs):
        """
        Apply regret matching
        :param obs: string that represents current state
        :return:
        """

        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.num_actions):
                action_probs[action] = 1.0 / self.env.num_actions
        return action_probs

    def action_probs(self, obs, legal_actions, policy):
        """
        Compute action probabilities given observation, legal actions, and policy
        :param obs: string that represents current state
        :param legal_actions: list of legal actions
        :param policy: dictionary of probabilities of actions that have been made
        :return:
        """

        if obs not in policy.keys():
            action_probs = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def get_state(self, player_id):
        """
        Get current state as a string
        :param player_id: player
        :return: state as string and indices of legal actions
        """

        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['legal_actions'].keys())

    def eval_step(self, state):
        """
        Prediction based on current state
        :param state: array that represents current state
        :return: predicted action and dictionary containing info
        """

        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(probs), p=probs)

        info = {'probs': {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in
                          range(len(state['legal_actions']))}}

        return action, info

    def save(self):
        """
        Save model
        """

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        """
        Load model
        """

        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()
