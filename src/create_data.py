from pypokerengine.api.game import setup_config, start_poker

from models.HumanPlayer import HumanPlayer
from models.TestPlayer import TestPlayer

config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=TestPlayer())
config.register_player(name="p2", algorithm=HumanPlayer())
game_result = start_poker(config, verbose=1)