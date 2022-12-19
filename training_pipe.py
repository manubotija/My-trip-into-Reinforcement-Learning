
from models import *
from game import Game, GameOptions
from wrappers import TorchWrapper
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

def run_training(options, model_class, hyperparams=None, reward_type=1, model_weights=None):
    env = TorchWrapper(Game(  render_mode=None, 
                                                options=options, 
                                                render_fps=1000,
                                                reward_type=reward_type,
                                                ), device=device, normalize=False) 
    eval_env = TorchWrapper(Game(  render_mode=None, 
                                                options=options, 
                                                render_fps=1000,
                                                reward_type=reward_type,
                                                ), device=device, normalize=False) 

    obs_shape, num_actions_act, action_shape = env.get_shapes()

    # policy_model is the model that is trained
    policy_model = model_class(obs_shape, num_actions_act).to(device)
    # target model is the model that is used to calculate the target Q values
    target_model = model_class(obs_shape, num_actions_act).to(device)

    if model_weights is not None:
        policy_model.load_state_dict(model_weights)
        target_model.load_state_dict(model_weights)

    optimizer = optim.Adam(policy_model.parameters(), lr=hyperparams.lr, amsgrad=True)

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
                                                        num_episodes = 1000,
                                                        target_avg_score=200)
    

    save_path = save_model( policy_model=policy_model,
                            target_model=target_model,
                                agent=agent,
                                options=options,
                                scores=rewards,
                                durations=durations,
                                loss=loss)

    return save_path, success, average_score, speed


def _init_table():
    # Create a DataFrame with column names specified
    table = pd.DataFrame(columns=['save_path','gamma', 'eps_decay', 'eps_end', 'batch_size', 'reward_type', 'average_score', 'speed', 'success'])
    return table

def _add_to_table(table, hyperParams, average_score, speed, save_path, reward_type, success):
    # Create a new dataframe with the new row
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


def hyperparameter_search(options, hyperParams):
# Hyperparameter search
    gamma_list = [0.99,0.9,0.999]
    eps_decay_list = [4000, 8000]
    eps_end_list = [0.1, 0.05]
    batch_size_list = [128]
    reward_type_list = [2,3,4]

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
                        for i in range(max_tries_per_config):
                            save_path, success, average_score, avg_speed = run_training(
                                                                        options, 
                                                                        SimpleLinearModel, 
                                                                        hyperParams, 
                                                                        reward_type)

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

options = GameOptions(  height=300, 
                        width=300, 
                        n_obstacles=0, 
                        n_turrets=0, 
                        max_projectiles_per_turret=0, 
                        fire_turret_step_delay=0,
                        max_steps=300
                        )

options.player_bounds = Rect(0, options.height*0.8, options.width, options.height*0.2)
options.gate_bounds= Rect(0, 0, options.width, options.height*0.2)

hyperparameter_search(options, hyperParams)
quit()
#weights_path, success, average_score = run_training(options, SimpleLinearModel, hyperParams)

model_path = "20221211-225256"

#plot_model_training_history(model_path)
_ , weights, _ = load_model(model_path)
weights_path, success, average_score = run_training(options, SimpleLinearModel, hyperParams, weights)

quit()

options.fire_turret_step_delay = 100
