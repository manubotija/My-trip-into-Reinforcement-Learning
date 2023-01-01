

import gym

import stable_baselines3 as sb3
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from gym.utils.env_checker import check_env as gym_check_env
from game import Game, GameOptions
from wrappers import *
import datetime

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
                    policy_kwargs=policy_kwargs
                    )
  return model

def get_PPO_model(env):
  
  policy_kwargs = dict(   
                      # activation_fn=torch.nn.ReLU, 
                      net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                      # optimizer_class = torch.optim.AdamW, 
                      # optimizer_kwargs = dict(amsgrad=True),
                      # normalize_images=False
                      )
  model = PPO("MlpPolicy", env, verbose=1,
                    # learning_rate=LR, 
                    # gamma=GAMMA, 
                    policy_kwargs=policy_kwargs
                    )
  return model

options = GameOptions()
options.max_steps = 300
options.max_projectiles_per_turret = 0
options.instantiate_obstacles = False
options.instantiate_turrets = False
options.reward_type = 4



env = Game(render_mode=None, options=options)
gym_check_env(env)
settings = TorchWrapperSettings(normalize=True, flatten_action=True, skip_frames=None, to_tensor=False)
env = TorchWrapper(env, wrapper_settings = settings)
gym_check_env(env)
check_env(env, warn=True)

model = get_PPO_model(env)
# model.learn(total_timesteps=200_000, progress_bar=True)
# model.save("sb3/dqn_baseline")
# quit()
# input("Press Enter to continue...")

# model = DQN.load("sb3/dqn_baseline", env=env)

env.options.instantiate_obstacles = True
env.options.instantiate_turrets = True
env.options.reward_type = 4

model.learn(total_timesteps=1_000_000, progress_bar=True)


# # #save model with current date
model.save("sb3/model_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
# model = PPO.load("sb3/model_20230101-225453", env=env)
input = input("Press Enter to continue...")
#if input is p, continue, else quite
if input == "p":
    env = Game(render_mode="human", options=options)
    env = TorchWrapper(env, wrapper_settings = settings)
    obs, _ = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()
        # VecEnv resets automatically
        if done:
          obs, _ = env.reset()

    env.close()