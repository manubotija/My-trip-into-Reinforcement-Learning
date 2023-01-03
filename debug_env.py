
from game import Game, GameOptions
from wrappers import GameWrapper, GameWrapperSettings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# create a game environment
options = GameOptions(
    width = 600,
    height = 600,
    instantiate_obstacles=False,
    instantiate_turrets=False
)
env = Game(render_mode=None, options=options)
settings = GameWrapperSettings(normalize=True, to_tensor=False, flatten_action=True)
env = GameWrapper(env, device='cpu', wrapper_settings=settings)

print (env.observation_space)

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
    #create list of 16 pre-defined colors
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan', 'magenta', 'lime', 'teal', 'maroon', 'navy']
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
            y1, x1, y2, x2 = obs[j:j+4]
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

game_obs_tester(env)