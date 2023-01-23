
from game import Game, GameOptions
from wrappers import GameWrapper, GameWrapperSettings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from settings import *
#import gym check_env
from gym.utils.env_checker import check_env as gym_check_env


def game_obs_tester(env):
    """ Method that takes an environment
        Then performs the following:
        1. Resets the environment
        2. Gets the observation space
        3. Initializes matplot figure
        4. Iterates for 1000 steps
            4.2. Creates rectangles to be renderd in matplot where every four elements in 
            the observation correspond to the top left corner and bottrom right corners of a square. Each rectangle is draw with a different color
            4.3  Live render the rectangles into the matplot figure
            4.3. Gets the action space
            4.4. Samples an action from the action space
            4.5. Performs the action

    """
    # reset the environment
    obs, _ = env.reset()
    # initialize the matplot figure
    fig, ax = plt.subplots()
    # set x and y axis limits to the size of the game
    if not settings.normalize:
        ax.set_xlim(0, env.options.width)
        ax.set_ylim(0, env.options.height)
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    #create list of 16 pre-defined colors
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan', 'magenta', 'lime', 'teal', 'maroon', 'navy']
    
    #draw one the bounds of the player, the gate, the turrets and the obstacles 
    # using a rectangle without fill but with a color border
    # remember the bounds are defined by the top left corner and the bottom right corner
    # and are an instance of Pygame's Rect class
    # first normalize the bounds to the range [-1, 1] and collect them under a list
    bounds = []
    if settings.normalize:
        gate_rect = Rectangle((env.options.gate_bounds.left*2/env.options.width-1, env.options.gate_bounds.top*2/env.options.height-1), env.options.gate_bounds.width*2/env.options.width, env.options.gate_bounds.height*2/env.options.height, edgecolor='black', facecolor='none', linewidth=2)
        obstacle_rect = Rectangle((env.options.obstacle_bounds.left*2/env.options.width-1, env.options.obstacle_bounds.top*2/env.options.height-1), env.options.obstacle_bounds.width*2/env.options.width, env.options.obstacle_bounds.height*2/env.options.height, edgecolor='teal', facecolor='none', linewidth=2)
        turret_rect = Rectangle((env.options.turret_bounds.left*2/env.options.width-1, env.options.turret_bounds.top*2/env.options.height-1), env.options.turret_bounds.width*2/env.options.width, env.options.turret_bounds.height*2/env.options.height, edgecolor='red', facecolor='none', linewidth=2)
        player_rect = Rectangle((env.options.player_bounds.left*2/env.options.width-1, env.options.player_bounds.top*2/env.options.height-1), env.options.player_bounds.width*2/env.options.width, env.options.player_bounds.height*2/env.options.height, edgecolor='yellow', facecolor='none', linewidth=2)
    else:
        gate_rect = Rectangle((env.options.gate_bounds.left, env.options.gate_bounds.top), env.options.gate_bounds.width, env.options.gate_bounds.height, edgecolor='black', facecolor='none', linewidth=2)
        obstacle_rect = Rectangle((env.options.obstacle_bounds.left, env.options.obstacle_bounds.top), env.options.obstacle_bounds.width, env.options.obstacle_bounds.height, edgecolor='teal', facecolor='none', linewidth=2)
        turret_rect = Rectangle((env.options.turret_bounds.left, env.options.turret_bounds.top), env.options.turret_bounds.width, env.options.turret_bounds.height, edgecolor='red', facecolor='none', linewidth=2)
        player_rect = Rectangle((env.options.player_bounds.left, env.options.player_bounds.top), env.options.player_bounds.width, env.options.player_bounds.height, edgecolor='yellow', facecolor='none', linewidth=2)

    # add the rectangles to the list of bounds
    bounds.append(gate_rect)
    bounds.append(obstacle_rect)
    bounds.append(turret_rect)
    bounds.append(player_rect)
    
    # iterate for 1000 steps
    for i in range(1000):
        # create rectangles to be renderd in matplot where every four elements in the observation correspond to the top left corner and bottrom right corners of a square. 
        # Each rectangle is draw with a different color
        rects = []
        print(obs)
        assert len(obs) == env.observation_space.shape[0]
        # remove all labels
        for text in ax.texts:
            text.remove()
        for j in range(0, len(obs)-1, 4):
            x1, y1, x2, y2 = obs[j:j+4]
            w = x2 - x1
            h = y2 - y1
            color = colors[(j//4)%len(colors)]
            rects.append(Rectangle((x1, y1), w, h, color=color)) 
            #label each rectangle with the index of the color
            ax.text(x1, y1, j//4, color='black')

            #print in the console the coordinates of the rectangle
            print(f"Rectangle {j//4} ({color}) coordinates: ({x1}, {y1}), ({x2}, {y2}), size: {w}x{h}")

        # live render the rectangles into the matplot figure
        for patch in ax.patches:
            patch.remove()
        for rect in bounds:
            ax.add_patch(rect)
        for rect in rects:
            ax.add_patch(rect)

        plt.draw()
        ax.set_title(f"Step: {i}")
        plt.pause(0.2)
        # get the action space
        action_space = env.action_space
        # sample an action from the action space
        action = action_space.sample()
        # perform the action
        obs, reward, done, _, info = env.step(action)
        #pause 5s
        
        if done:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    # create a game environment
    options = GameOptions.from_yaml('./game_scenarios.yaml', '800x800-30o-2t-no-proj')
    env = Game(render_mode=None, options=options)
    settings = GameWrapperSettings(normalize=False, to_tensor=False, flatten_action=True)
    env = GameWrapper(env, device='cpu', wrapper_settings=settings)
    print (env.observation_space)
    obs, _ = env.reset()
    print (obs)
    gym_check_env(env)
    game_obs_tester(env)