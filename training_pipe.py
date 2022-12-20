
from models import *
from game import Game, GameOptions
from wrappers import TorchWrapper, TorchWrapperSettings
from gym.wrappers import FlattenObservation
from agent import Agent, HyperParams
import numpy as np
import matplotlib.pyplot as plt
import time
from pygame import Rect
from utils import *
import sys
import pandas as pd

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(options, model_class, wrapper_settings, hyperparams=None, model_weights=None):
    env = TorchWrapper(Game(  render_mode=None, 
                                                options=options, 
                                                render_fps=1000,
                                                ), device=device, 
                                                wrapper_settings=wrapper_settings) 
    eval_env = TorchWrapper(Game(  render_mode=None, 
                                                options=options, 
                                                render_fps=1000,
                                                ), device=device,
                                                wrapper_settings=wrapper_settings) 

    obs_shape, num_actions_act, action_shape = env.get_shapes()

    model_params = ModelParams(obs_shape, num_actions_act, 256, 4)

    # policy_model is the model that is trained
    policy_model = model_class(model_params).to(device)
    # target model is the model that is used to calculate the target Q values
    target_model = model_class(model_params).to(device)

    if model_weights is not None:
        policy_model.load_state_dict(model_weights)
        target_model.load_state_dict(model_weights)

    optimizer = optim.AdamW(policy_model.parameters(), lr=hyperparams.lr, amsgrad=True)

    agent = Agent(policy_model=policy_model, 
                        target_model=target_model, 
                        env=env, 
                        optimizer=optimizer, 
                        action_shape=action_shape, 
                        device=device,
                        hyperparams=hyperparams,
                        eval_env=eval_env,
                        live_plot=False,
                        final_plot=False)
    
    durations, rewards, success, loss, average_score, speed = agent.do_train(
                                                        num_episodes = 3000,
                                                        target_avg_score=100000)
    

    save_path = save_model( policy_model=policy_model,
                            target_model=target_model,
                                agent=agent,
                                options=options,
                                scores=rewards,
                                durations=durations,
                                loss=loss, wrapper_settings=wrapper_settings)

    return save_path, success, average_score, speed


def _init_table():
    # Create a DataFrame with column names specified
    table = pd.DataFrame(columns=['save_path','gamma', 'eps_decay', 'eps_end', 'batch_size', 'reward_type', 'average_score', 'speed', 'success'])
    return table

def _add_to_table(table, hyperParams, average_score, speed, save_path, reward_type, success):
    # Create a new dataframe with the new row
    success = bool(success)
    new_row = pd.DataFrame({'save_path': save_path,
                        'gamma': hyperParams.gamma,
                        'eps_decay': hyperParams.eps_decay,
                        'eps_end': hyperParams.eps_end,
                        'batch_size': hyperParams.batch_size,
                        'reward_type': reward_type,
                        'average_score': average_score,
                        'speed': speed, 
                        'success': success}, index=[0])
    # Concatenate the new row to the existing dataframe
    table = pd.concat([table, new_row], ignore_index=True)
    return table


def hyperparameter_search(options, hyperParams, wrapper_settings):
# Hyperparameter search
    gamma_list = [0.98]
    eps_decay_list = [4000]
    eps_end_list = [0.05]
    batch_size_list = [128]
    reward_type_list = [4]

    max_tries_per_config = 1

    #iterate over all configs
    best_average_score = 0
    best_speed = sys.maxsize
    best_hyperparams = None
    best_options = None
    table = _init_table()

    for batch_size in batch_size_list:
        for gamma in gamma_list:
            for eps_decay in eps_decay_list:
                for eps_end in eps_end_list:
                    for reward_type in reward_type_list:
                        hyperParams.gamma = gamma
                        hyperParams.eps_decay = eps_decay
                        hyperParams.eps_end = eps_end
                        hyperParams.batch_size = batch_size
                        options.reward_type = reward_type
                        for i in range(max_tries_per_config):
                            save_path, success, average_score, avg_speed = run_training(
                                                                        options,
                                                                        SimpleLinearModel,
                                                                        wrapper_settings, 
                                                                        hyperParams)

                            table = _add_to_table(table, hyperParams, average_score, avg_speed, save_path, reward_type, success)
                            print(table)

                            if average_score > best_average_score:
                                best_average_score = average_score
                                best_speed = avg_speed
                                best_hyperparams = hyperParams
                                best_options = options
                                best_reward_type = reward_type
                                print("=====New best score ", average_score, "with speed", avg_speed, "======")
                            if success: 
                                print("=====Success! ======")
                                print(table)
                                return

    print("==============================================")
    print("Failed to solve task with any config")
    print("Best config: ", best_hyperparams)
    print("with options: ", options)
    print("with reward_type: ", best_reward_type)
    print("resulted in average speed: ",best_speed, " and average_score: ", best_average_score, "======")
    print("Table of results: ")
    print(table)
    print("==============================================")
    return


wrapper_settings = TorchWrapperSettings(
                        normalize=True, 
                        flatten_action=True)


hyperParams = HyperParams(
                        memory_size=10000, 
                        batch_size=None, 
                        gamma=None, 
                        tau=0.005,
                        eps_start=0.95,
                        eps_end=None,
                        eps_decay=None,
                        lr=1e-4
                        )

# options = GameOptions(  height=600, 
#                         width=400, 
#                         n_obstacles=1, 
#                         n_turrets=1, 
#                         max_projectiles_per_turret=3, 
#                         fire_turret_step_delay=30,
#                         max_steps=300,
#                         reward_type=None,
#                         )

# options.player_bounds = Rect(0, 0, options.width, options.height)
# options.gate_bounds= Rect(0, 0, options.width, options.height)

options = GameOptions()
options.max_steps = 300

hyperparameter_search(options, hyperParams, wrapper_settings)