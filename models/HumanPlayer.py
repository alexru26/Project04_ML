from src.util.bluff_classifier import is_bluff
from src.util.landmark import get_landmark
from pypokerengine.players import BasePokerPlayer
import pypokerengine.utils.visualize_utils as U

def get_record():
    res = ''
    while res not in ['y', 'n']:
        res = input("Record data? >> ")
    return res == 'y'

class HumanPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        self.record = get_record()
        self.bluff_counter, self.nonbluff_counter = 0, 0
        if self.record: self.get_count()

    def get_count(self):
        with open('../data/count.txt', 'r') as f:
            text = f.readlines()

        self.bluff_counter = int(text[0].split(' ')[1])
        self.nonbluff_counter = int(text[1].split(' ')[1])

    def update_count(self):
        with open('../data/count.txt', 'w') as f:
            f.write('bluff: ' + str(self.bluff_counter) + '\n' + 'nonbluff: ' + str(self.nonbluff_counter))

    def declare_action(self, valid_actions, hole_card, round_state):
        print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))
        action, amount = self._receive_action_from_console(valid_actions)
        if self.record:
            if is_bluff(hole_card, round_state['community_card']):
                get_landmark(True, self.bluff_counter)
                self.bluff_counter += 1
                self.update_count()
            else:
                get_landmark(False, self.nonbluff_counter)
                self.nonbluff_counter += 1
                self.update_count()
        return action, amount

    def receive_game_start_message(self, game_info):
        print(U.visualize_game_start(game_info, self.uuid))
        self._wait_until_input()

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(U.visualize_round_start(round_count, hole_card, seats, self.uuid))
        self._wait_until_input()

    def receive_street_start_message(self, street, round_state):
        print(U.visualize_street_start(street, round_state, self.uuid))
        self._wait_until_input()

    def receive_game_update_message(self, new_action, round_state):
        print(U.visualize_game_update(new_action, round_state, self.uuid))
        self._wait_until_input()

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))
        self._wait_until_input()

    def _wait_until_input(self):
        input("Enter some key to continue...")

    def _receive_action_from_console(self, valid_actions):
        action = ""
        amount = 0
        while not (action == 'fold' or action == 'call' or action == 'raise'):
            action = input("Enter action to declare >> ")
        if action == 'fold': amount = 0
        if action == 'call': amount = valid_actions[1]['amount']
        if action == 'raise':
            amount = 0
            while not (amount > 0):
                try:
                    amount = int(input("Enter raise amount >> "))
                except ValueError:
                    print("Invalid input")
        return action, amount