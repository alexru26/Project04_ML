import rlcard
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.utils import print_card

from models.cfr_model.CFR_Model import CFRModel
from models.deep_cfr_model.Deep_CFR_Model import DeepCFRModel
from models.face_deep_cfr_model.Face_Deep_CFR_Model import FaceDeepCFRModel
from src.util.landmark import get_landmark

global env

def setup():
    """
    Setup environment and agent
    :return: Exception if invalid model
    """

    global env

    # Make environment
    env = rlcard.make(environment)

    # Define human agent
    human_agent = HumanAgent(env.num_actions)

    # Load corresponding model
    if model == 'cfr_model':
        agent = CFRModel(env=env, model_path='../models/cfr_model/'+str(environment)+'/cfr_model')
    elif model == 'deep_cfr_model':
        agent = DeepCFRModel(env=env, model_path='../models/deep_cfr_model/'+str(environment)+'/deep_cfr_model')
    elif model == 'face_deep_cfr_model':
        agent = FaceDeepCFRModel(env=env, model_path='../models/face_deep_cfr_model/'+str(environment)+'/face_deep_cfr_model')
    elif model == 'human':
        agent = HumanAgent(env.num_actions)
    else:
        return Exception('Invalid model')

    # Set agents to environment
    env.set_agents([human_agent, agent])

def main():
    """
    Main game loop
    """

    while True:
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False) # Run the game

        # =============== Game Over =============== #
        final_state = trajectories[0][-1] # Get final state
        action_record = final_state['action_record'] # Get record of actions
        state = final_state['raw_obs'] # Get observation of final state
        _action_list = [] # List of player and action pair throughout game
        for i in range(1, len(action_record)+1): # For every player and action pair in action record
            if action_record[-i][0] == state['current_player']:
                break
            _action_list.insert(0, action_record[-i]) # Insert action
        for pair in _action_list: # For every player and action pair
            print('>> Player', pair[0], 'chooses', pair[1]) # Print what happened

        print('===============     Cards all Players    ===============')
        for hands in env.get_perfect_information()['hand_cards']: # For every player
            print_card(hands) # Show hand

        print('===============     Result     ===============')
        if payoffs[0] > 0: # Win
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0: # Tie
            print('It is a tie.')
        else: # Lose
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        input("Press any key to continue...")

if __name__ == '__main__':
    """
    Main entry point
    """

    environment = 'leduc-holdem'
    model = 'human'

    setup()
    main()

    # get_landmark(True, 1)
