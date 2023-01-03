from game import Game, GameOptions
from sprites import Bounds

options = GameOptions(height=300, width=300)
options.max_steps = 300
options.max_projectiles_per_turret = 0
options.instantiate_obstacles = False
options.instantiate_turrets = False
options.reward_type = 4
#place the player in 10% to the bottom of the screen and the get 10% to the top. Both centered horizontally
options.player_bounds = Bounds(130,250, 40, 50)
options.gate_bounds = Bounds(120,0, 60, 60)
game = Game(render_mode='human', options=options)
game.reset()
game.run_loop()