

import gym

import stable_baselines3 as sb3
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from gym.utils.env_checker import check_env as gym_check_env
from game import Game, GameOptions
from wrappers import *
import datetime
from sprites import Bounds
import imageio
from evaluate import evaluate, create_env

MEM_SIZE = 10_000
BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 4000 # The higher the longer it takes to decay
TAU = 0.005
LR = 1e-4

def get_DQN_model(env):
  policy_kwargs = dict(   
                        # activation_fn=torch.nn.ReLU, 
                        net_arch=[128, 128, 128], 
                        # optimizer_class = torch.optim.AdamW, 
                        # optimizer_kwargs = dict(amsgrad=True),
                        # normalize_images=False
                        )
  model = DQN("MlpPolicy", env, verbose=1,
                    # learning_rate=LR, 
                    # buffer_size=MEM_SIZE, 
                    # batch_size=BATCH_SIZE, 
                    # gamma=GAMMA, 
                    # tau=TAU, 
                    # exploration_initial_eps=EPS_START, 
                    # exploration_final_eps=EPS_END, 
                    # train_freq=1,
                    # learning_starts=BATCH_SIZE,
                    # target_update_interval=1,
                    # gradient_steps=-1,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./tensorboard/"
                    )
  return model

def get_PPO_model(env):
  
  policy_kwargs = dict(   
                      # activation_fn=torch.nn.ReLU, 
                      net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                      # optimizer_class = torch.optim.AdamW, 
                      # optimizer_kwargs = dict(amsgrad=True),
                      # normalize_images=False
                      )
  model = PPO("MlpPolicy", env, verbose=1,
                    # learning_rate=LR, 
                    # gamma=GAMMA, 
                    gae_lambda=0.99,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./tensorboard/"
                    )
  return model

if __name__ == '__main__':

  options = GameOptions()
  options.max_steps = 300
  options.max_projectiles_per_turret = 0
  options.instantiate_obstacles = True
  options.instantiate_turrets = True
  # options.player_bounds = Bounds(130,250, 40, 50)
  # options.gate_bounds = Bounds(120,0, 60, 60)

  settings = GameWrapperSettings(normalize=True, flatten_action=True, skip_frames=None, to_tensor=False)

  env = create_env(options, settings, vectorized=True, monitor=True, render_mode=None)
  eval_env = create_env(options, settings, vectorized=False, monitor=True, render_mode=None)
  model = get_PPO_model(env)

  RUN_NAME = "PP0_new_rewards_inst_coll-3"
  logs_path = "./eval_logs/{}".format(RUN_NAME)
  eval_callback = EvalCallback(eval_env, best_model_save_path=logs_path,
                              log_path=logs_path, eval_freq=50_000, n_eval_episodes=100,
                              deterministic=False, render=False)

  model.learn(total_timesteps=1_000_000, progress_bar=True, callback=eval_callback, tb_log_name=RUN_NAME)
  env.close()
  eval_env.close()

  print("Eval last mean reward: {:.2f}".format(eval_callback.last_mean_reward))
  print("Eval best mean reward: {:.2f}".format(eval_callback.best_mean_reward))

  evaluate(model, options, settings, deterministic=True, n_episodes=10, save_gif=True)
  evaluate(model, options, settings, deterministic=False, n_episodes=10, save_gif=True)
  evaluate(model, options, settings, deterministic=True, n_episodes=1000, save_gif=False)
  evaluate(model, options, settings, deterministic=False, n_episodes=1000, save_gif=False)