

import gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from gym.utils.env_checker import check_env as gym_check_env
from game import Game, GameOptions
from wrappers import *
import datetime
from settings import *
from evaluate import evaluate, create_env, get_best_length, get_max_min_reward
import argparse

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

def get_PPO_model(env, tensorboard_path):
  
  policy_kwargs = dict(   
                      activation_fn=torch.nn.Tanh, 
                      net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                      
                      # optimizer_class = torch.optim.AdamW, 
                      # optimizer_kwargs = dict(amsgrad=True),
                      # normalize_images=False
                      )
  model = PPO("MlpPolicy", env, verbose=1,
                    tensorboard_log=tensorboard_path,
                    clip_range=0.3,
                    ent_coef=0.0019694798370554207,
                    gae_lambda=0.9,
                    gamma=0.9997,
                    learning_rate=1.2188370022818374e-05,
                    max_grad_norm=0.8,
                    n_epochs=5,
                    n_steps=1024,
                    vf_coef=0.5709964204264004,
                    policy_kwargs = policy_kwargs
                    )
  return model


def train(args):

  GAME_SCENARIO = args.scenario
  REWARD_SCHEME = args.reward
  model_path = args.model_path

  options = GameOptions.from_yaml("configs/game_scenarios.yaml", GAME_SCENARIO)
  rewards = RewardScheme.from_yaml("configs/rewards.yaml", REWARD_SCHEME)
  options.rew = rewards.normalize(rewards.win_reward)
  settings = GameWrapperSettings(normalize=True, flatten_action=True, skip_frames=args.skip_frames, to_tensor=False)

  env = create_env(options, settings, vectorized=False, monitor=True, render_mode=None, n_envs=args.n_envs)
  eval_env = create_env(options, settings, vectorized=True, monitor=True, render_mode=None)
  
  LOG_PATH = "./logs/" + args.project_name + "/" + GAME_SCENARIO + "_" + REWARD_SCHEME + "_{}/".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
  if model_path is None:
    # Create a new model
    model = get_PPO_model(env, LOG_PATH)
  else:
    #Load a model
    model = PPO.load(model_path, env=env, tensorboard_log=LOG_PATH)

  eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_PATH,
                              log_path=LOG_PATH, eval_freq=100_000, n_eval_episodes=300,
                              deterministic=False, render=False)
  
  model.learn(total_timesteps=args.time_steps, progress_bar=True, callback=eval_callback)

  env.close()
  eval_env.close()

  print("----RESULTS FOR {}".format(LOG_PATH + "----"))
  print("Eval last mean reward: {:.2f}".format(eval_callback.last_mean_reward))
  print("Eval best mean reward: {:.2f}".format(eval_callback.best_mean_reward))
  best_model_path = eval_callback.best_model_save_path + 'best_model.zip'
  print("Best model saved to: " + best_model_path)

  # Save GIF for the best model
  model = PPO.load(best_model_path)
  evaluate(model, options, settings, deterministic=True, n_episodes=10, save_gif_path=LOG_PATH)
  evaluate(model, options, settings, deterministic=False, n_episodes=10, save_gif_path=LOG_PATH)

  if args.extra_eval:
    best_length_mean, best_length_std, fastest_mean_reward, fastest_std_reward = get_best_length(eval_callback)
    print("Fastest eval: {:.2f} steps +/- {:.2f}".format(best_length_mean, best_length_std))
    print("Fastest mean reward: {:.2f} +/- {:.2f}".format(fastest_mean_reward, fastest_std_reward))
    get_max_min_reward(eval_callback)
    
    evaluate(model, options, settings, deterministic=True, n_episodes=1000, save_gif_path=None)
    evaluate(model, options, settings, deterministic=False, n_episodes=1000, save_gif_path=None)

def add_subarguments(parser):
  parser.add_argument('scenario', type=str, help='Name of game scenario, from configs/game_scenarios.yaml')
  parser.add_argument('reward', type=str, help='Name of reward scheme, from configs/rewards.yaml')
  parser.add_argument('--model_path', type=str, help='Path to model to load')
  parser.add_argument('--time_steps', type=int, help='Number of time steps to train for', default=1_000_000)
  parser.add_argument('--extra_eval', action='store_true', help='Evaluate the model on 1000 episodes')
  parser.add_argument('--skip_frames', type=int, help='Number of frames to skip', default=4)
  parser.add_argument('--n_envs', type=int, help='Number of environments to run in parallel', default=6)
  parser.add_argument('--project_name', type=str, help='Name a project to have all logs stored in the same folder over multiple experiments', default='')