
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from wrappers import GameWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from old_code.agent import Agent, HyperParams
from old_code.models import SimpleLinearModel

if gym.__version__[:4] == '0.26':
    env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')
elif gym.__version__[:4] == '0.25':
    env = gym.make('CartPole-v1', new_step_api=True)
    eval_env = gym.make('CartPole-v1', new_step_api=True)
else:
    raise ImportError(f"Requires gym v25 or v26, actual version: {gym.__version__}")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# Get number of actions from gym action space
n_actions = env.action_space.n
action_space_shape = (n_actions)
action_shape = (1,n_actions)
# Get the number of state observations
if gym.__version__[:4] == '0.26':
    state, _ = env.reset()
elif gym.__version__[:4] == '0.25':
    state, _ = env.reset(return_info=True)
n_observations = len(state)

env = GameWrapper(env, normalize=False)
eval_env = None

policy_model = SimpleLinearModel(n_observations, n_actions).to(device)
target_model = SimpleLinearModel(n_observations, n_actions).to(device)
target_model.load_state_dict(policy_model.state_dict())

optimizer = optim.AdamW(policy_model.parameters(), lr=LR, amsgrad=True)

hyperParams = HyperParams(
                        memory_size=10000, 
                        batch_size=BATCH_SIZE, 
                        gamma=GAMMA, 
                        tau=TAU,
                        eps_start=EPS_START,
                        eps_end=EPS_END,
                        eps_decay=EPS_DECAY,
                        lr=LR,
                        )

agent = Agent(policy_model=policy_model, 
                    target_model=target_model, 
                    env=env, 
                    optimizer=optimizer, 
                    action_shape=action_shape, 
                    device=device,
                    hyperparams=hyperParams,
                    eval_env=eval_env,
                    live_plot=False,
                    final_plot=True)

durations, rewards, success, loss, average_score, speed = agent.do_train(
                                                    num_episodes = 1000,
                                                    target_avg_score=500)

print('Complete')
