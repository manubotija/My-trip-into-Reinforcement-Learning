from game import Game, GameOptions
from settings import *


options = GameOptions.from_yaml("configs/game_scenarios.yaml", "800x800-mid_barrier-no-proj")
rewards = RewardScheme.from_yaml("configs/rewards.yaml", "config_3")
options.rew = rewards.normalize(rewards.win_reward)
game = Game(render_mode="human", options=options)
game.reset()
game.run_loop()

