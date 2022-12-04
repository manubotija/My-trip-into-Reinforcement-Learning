from game import Game
from gym.utils.env_checker import check_env
import numpy as np

def run_env(env, num_episodes=100, render=False):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            # action = env.action_space.sample()
            action = np.random.randint(0, 3, size=2)
            obs, reward, done, _, info = env.step(action)
            if render:
                env.render()
        print("Episode finished. Reward: %s, Score:%s" % (reward, info['score']))
    env.close()
def test_env_no_render():
    env = Game(render_mode=None)
    check_env(env)
    run_env(env, num_episodes=10, render=False)

def test_env_human_render():
    env = Game(render_mode='human', render_fps=60)
    check_env(env)
    run_env(env, num_episodes=10, render=True)

def test_env_rgb_render():
    env = Game(render_mode="rgb_array", render_fps=60)
    check_env(env)
    env.reset()
    screen = env.render()
    assert screen.shape == (env.options.height, env.options.width, 3)
    run_env(env, num_episodes=10, render=True)