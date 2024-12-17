import os
import argparse

import rlcard
from rlcard.agents import (
    RandomAgent,
)
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)

from models.cfr_model.CFR_Model import CFRModel
from models.deep_cfr_model.Deep_CFR_Model import DeepCFRModel
from models.face_deep_cfr_model.Face_Deep_CFR_Model import FaceDeepCFRModel

from timeit import default_timer as timer

global env, eval_env, agent

def setup():
    """
    Setup environment and agent
    :return: Exception if invalid model
    """

    global env, eval_env, agent

    # Make environment and evaluation environment
    env = rlcard.make(
        environment,
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        environment,
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random, etc
    set_seed(args.seed)

    # Load corresponding model
    if model == 'cfr_model':
        agent = CFRModel(
            env,
            os.path.join(
                args.log_dir,
                'cfr_model',
            ),
        )
    elif model == 'deep_cfr_model':
        agent = DeepCFRModel(
            env,
            os.path.join(
                args.log_dir,
                'deep_cfr_model',
            )
        )
    elif model == 'face_deep_cfr_model':
        agent = FaceDeepCFRModel(
            env,
            os.path.join(
                args.log_dir,
                'face_deep_cfr_model',
            )
        )
    else:
        return Exception('Invalid model')

    # Set agents to evaluation environment
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

def train():
    """
    Main train function
    """

    print("Start training")
    with Logger(args.log_dir) as logger: # Log information
        start = timer() # Timer to keep track of time took to train
        for episode in range(args.num_episodes): # Train for number of episodes
            agent.train() # Train model
            print('\rIteration {}'.format(episode), end='')

            # Evaluate performance
            if episode % args.evaluate_every == 0:
                agent.save() # Save model first
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )

        # Get paths for logs
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot learning curve
    plot_curve(csv_path, fig_path, model)

    print(f"Duration: {timer()-start}")
    print(f"Total iterations: {agent.iteration}")

if __name__ == '__main__':
    """
    Main entry point
    """

    environment = 'leduc-holdem'
    model = 'cfr_model'

    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../models/'+str(model)+'/'+str(environment),
    )
    args = parser.parse_args()

    setup()
    train()