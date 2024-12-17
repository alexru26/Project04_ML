from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

def is_bluff():
    rate = 0 # TODO: Add some way to detect if bluffing
    print('Winning rate: '+ str(rate))
    return rate < 0.5