from game import Game, GameOptions
from settings import *
import train
import evaluate
import argparse

def play(args):
    options = GameOptions.from_yaml("configs/game_scenarios.yaml", args.scenario)
    rewards = RewardScheme.from_yaml("configs/rewards.yaml", args.reward)
    options.rew = rewards.normalize(rewards.win_reward)
    game = Game(render_mode="human", options=options)
    game.reset()
    game.run_loop()

"""
Running this file allows to do multiple options based on the first argument:
    - train: train a model
    - evaluate: evaluate a model
    - play: play a game as human
    - optimize: optimize hyperparameters

Together with the first argument, there are two mandatory arguments:
    - scenario: the game scenario to use
    - reward: the reward scheme to use

In addition, there additional arguments that depend on the first argument.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on a game scenario and reward scheme')

    subparsers = parser.add_subparsers( title='subcommands',
                                        description='valid subcommands', required=True, dest='subcommand')
    parser_train = subparsers.add_parser('train', help='Train a model')
    train.add_subarguments(parser_train)
    parser_train.set_defaults(func=train.train)
    
    parser_play = subparsers.add_parser('play', help='Play a game')
    parser_play.add_argument('scenario', type=str, help='Name of game scenario, from configs/game_scenarios.yaml')
    parser_play.add_argument('reward', type=str, help='Name of reward scheme, from configs/rewards.yaml')
    parser_play.set_defaults(func=play)

    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate a model')
    evaluate.add_subarguments(parser_evaluate)
    parser_evaluate.set_defaults(func=evaluate.evaluate)

    args = parser.parse_args()
    args.func(args)






