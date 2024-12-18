import numpy as np
import collections

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim

from rlcard.utils.utils import *

class DeepCFRModel:
    """
    Implementation of Deep CFR algorithm
    """

    def __init__(self, env, model_path, lr=0.001):
        """
        :param env: game environment
        :param model_path: path to save/load model
        :param lr: learning rate for neural networks
        """

        self.use_raw = False

        self.env = env
        self.model_path = model_path
        self.iteration = 0

        # Neural networks for policy and regrets
        input_dim = self.env.state_shape[0][0]  # State feature size
        num_actions = self.env.num_actions

        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)  # Outputs action probabilities
        )
        self.regret_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)  # Outputs regrets for each action
        )

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.regret_optimizer = optim.Adam(self.regret_network.parameters(), lr=lr)

        # Buffers for training
        self.policy_memory = []  # [(state, action_probs)]
        self.regret_memory = []  # [(state, regrets)]

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

        # Train the policy and regret networks using collected data
        self.train_network(self.policy_network, self.policy_optimizer, self.policy_memory, is_policy=True)
        self.train_network(self.regret_network, self.regret_optimizer, self.regret_memory, is_policy=False)

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

        # If game is over, return utilities
        if self.env.is_over():
            try:
                return self.env.get_payoffs()
            except Exception:
                return np.zeros(self.env.num_players)

        current_player = self.env.get_player_id()
        state, legal_actions = self.get_state(current_player)

        action_probs = self.action_probs(state, legal_actions)
        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)

        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            # Traverse to the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if current_player != player_id:
            return state_utility

        # Compute regret
        counterfactual_prob = (np.prod(probs[:current_player]) *
                               np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]
        regrets = np.zeros(self.env.num_actions)
        for action in legal_actions:
            regrets[action] = counterfactual_prob * (action_utilities[action][current_player]
                                                     - player_state_utility)

        # Store regrets and policy
        self.regret_memory.append((state, regrets))
        self.policy_memory.append((state, action_probs))

        return state_utility

    def train_network(self, network, optimizer, memory, is_policy):
        """
        Train the neural network
        :param network: policy or regret network
        :param optimizer: optimizer for neural network
        :param memory: stored data
        :param is_policy: whether the network is policy network
        """

        batch_size = 32
        data_size = len(memory)
        num_batches = max(data_size // batch_size, 1)

        for _ in range(num_batches):
            batch = [memory[i] for i in np.random.choice(data_size, min(data_size, batch_size), replace=False)]

            states, targets = zip(*batch)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

            optimizer.zero_grad()
            outputs = network(states_tensor)

            # Use MSE loss for regrets, and cross-entropy loss for policy
            if is_policy:
                loss = nn.CrossEntropyLoss()(outputs, targets_tensor)
            else:
                loss = nn.MSELoss()(outputs, targets_tensor)

            loss.backward()
            optimizer.step()

    def action_probs(self, obs, legal_actions):
        """
        Predict action probabilities given state, legal actions
        :param obs: current state
        :param legal_actions: list of legal actions
        :return:
        """

        state_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor).numpy()
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def get_state(self, player_id):
        """
        Get current state as a string
        :param player_id: player
        :return: state as string and indices of legal actions
        """

        state = self.env.get_state(player_id)
        return state['obs'], list(state['legal_actions'].keys())

    def eval_step(self, state):
        """
        Prediction based on current state
        :param state: array that represents current state
        :return: predicted action and dictionary containing info
        """

        action_probs = self.action_probs(state['obs'], list(state['legal_actions'].keys()))
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, {'probs': action_probs}

    def save(self):
        """
        Save model
        """

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        torch.save(self.policy_network.state_dict(), f"{self.model_path}/policy_network.pt")
        torch.save(self.regret_network.state_dict(), f"{self.model_path}/regret_network.pt")

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'), 'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        """
        Load model
        """

        if not os.path.exists(self.model_path):
            return

        self.policy_network.load_state_dict(torch.load(f"{self.model_path}/policy_network.pt", weights_only=True))
        self.regret_network.load_state_dict(torch.load(f"{self.model_path}/regret_network.pt", weights_only=True))

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'), 'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()
