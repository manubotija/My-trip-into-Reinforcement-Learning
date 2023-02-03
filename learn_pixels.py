from stable_baselines3 import PPO
from settings import *
from wrappers import *
from game import Game
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


if __name__ == "__main__":
        
    options = GameOptions.from_yaml("configs/game_scenarios.yaml", "mid-barrier-no-proj")
    rewards = RewardScheme.from_yaml("configs/rewards.yaml", "config-4")
    options.rew = rewards.normalize(rewards.win_reward)
    settings = GameWrapperSettings(normalize=False, flatten_action=True, skip_frames=8)

    env = SubprocVecEnv([lambda: ImageWrapper(GameWrapper(Game(options=options, render_mode="rgb_array"), settings)) for i in range(6)],
                            #start_method='spawn'
                            )
    env = VecMonitor(env, info_keywords=("is_success",))

    # env = ImageWrapper(GameWrapper(Game(options=options, render_mode="rgb_array"), settings), factor=8)
    # obs, info = env.reset()
    # print(obs.shape)
    # print(obs.min(), obs.max())
    # plt.imshow(obs, cmap="gray")

    # loop over env, doing random actions and displaying the observation and the last action
    # for i in range(100):
    #     action = env.action_space.sample()
    #     obs, reward, done, truncated, info = env.step(action)
    #     plt.imshow(obs)
    #     plt.show()
    #     plt.title("action: {}".format(action))
    #     plt.pause(0.5)
    #     if done or truncated:
    #         obs, info = env.reset()

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="temp/ppo_pixels")
    model.learn(total_timesteps=10_000_000, progress_bar=True)
    env.close()