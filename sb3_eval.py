import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from game import Game, GameOptions
from wrappers import *


MODEL_PATH = "./modelzoo/best_mode_gae99l.zip"
# MODEL_PATH = "./sb3/model_20230102-141538.zip"

model = sb3.PPO.load(MODEL_PATH)

options = GameOptions()
options.max_steps = 300
options.max_projectiles_per_turret = 0
options.instantiate_obstacles = True
options.instantiate_turrets = True
options.reward_type = 4

env = Game(render_mode="human", options=options)
settings = GameWrapperSettings(normalize=True, flatten_action=True, skip_frames=None, to_tensor=False)
env = GameWrapper(env, wrapper_settings = settings)
env = Monitor(env)

episode_rewards, episode_durations = evaluate_policy(model, env, n_eval_episodes=1000, deterministic=True, render=True, return_episode_rewards=True)

print(f"{len(episode_rewards)} episodes evaluated")
print(f"mean_rew = {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
print(f"mean_len = {np.mean(episode_durations):.2f} +/- {np.std(episode_durations):.2f}")

env.close()