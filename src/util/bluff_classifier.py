from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

def is_bluff(hole_card, community_card):
    rate = estimate_hole_card_win_rate(
        nb_simulation=1000,
        nb_player=3,
        hole_card=gen_cards(hole_card),
        community_card=gen_cards(community_card)
    )
    print('Winning rate: '+ str(rate))
    return rate < 0.5