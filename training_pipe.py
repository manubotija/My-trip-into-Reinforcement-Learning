
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
import copy

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
                                                        num_episodes = 500,
                                                        target_avg_score=100000)
    

    save_path = save_model( policy_model=policy_model,
                            target_model=target_model,
                                agent=agent,
                                options=options,
                                scores=rewards,
                                durations=durations,
                                loss=loss, wrapper_settings=wrapper_settings)

    return save_path, success, average_score, speed


def _init_table(hyperParams, options, wrapper_settings):
    """creates a table with all the dict keys of input arguments as columns"""
    table = pd.DataFrame(columns=[])
    table['save_path'] = None
    for key in hyperParams.__dict__.keys():
        table[key] = None
    for key in options.__dict__.keys():
        table[key] = None
    for key in wrapper_settings.__dict__.keys():
        table[key] = None 
    table['avg_score'] = None
    table['avg_speed'] = None
    table['success'] = None
    return table

def _add_to_table(table, hyperParams, options, wrapper_settings, average_score, speed, save_path, success):
    # Create a new row for the table
    # it adds the values of the hyperparameters, options and warappers to the new row
    success = bool(success)
    new_row = pd.DataFrame(columns=[], index=[0])
    new_row['save_path'] = save_path
    for key in hyperParams.__dict__.keys():
        new_row[key] = hyperParams.__dict__[key]
    for key in options.__dict__.keys():
        new_row[key] = options.__dict__[key]
    for key in wrapper_settings.__dict__.keys():
        new_row[key] = wrapper_settings.__dict__[key]
    new_row['avg_score'] = average_score
    new_row['avg_speed'] = speed
    new_row['success'] = success
    table = pd.concat([table, new_row], ignore_index=True)
    return table 

def print_reduced_table(table):
    """removes the columns that have the same value for all rows"""
    reduced_table = table.copy()
    for col in table.columns:
        if len(table[col].unique()) == 1:
            reduced_table = reduced_table.drop(col, axis=1)
    print(reduced_table)


def hyperparameter_search(options, hyperParams, wrapper_settings):

    max_tries_per_config = 1
    table = _init_table(hyperParams, options, wrapper_settings)

    for option in get_next_hyperparameter(options):
        for hyper_param in get_next_hyperparameter(hyperParams):
            for wrapper_setting in get_next_hyperparameter(wrapper_settings):
                for i in range(max_tries_per_config):
                    save_path, success, average_score, avg_speed = run_training(
                                                                option,
                                                                SimpleLinearModel,
                                                                wrapper_setting, 
                                                                hyper_param)

                    table = _add_to_table(table, hyper_param, option, wrapper_setting, average_score, avg_speed, save_path, success)
                    print(table)

                    if success: 
                        print("=====Success! ======")
                        print_reduced_table(table)
                        return

    print("==============================================")
    print("Search complete. Result summary: ")
    print_reduced_table(table)
    print("==============================================")
    return

def get_next_hyperparameter(input_set):
    """
    Method that automatically iterates over the input. 
    The input is expected to be instances of a class that containes the parameters to be iterated over.
    Whenever it finds a parameter that is a list it creates
    as many copies of the input set as there are elements in the list
    and replaces the parameter with the corresponding element in the list.
    It yields the new input set for each iteration.
    """
    output_set = [input_set]
    for_removal = []
    for set in output_set:
        for key in set.__dict__:
            value = set.__dict__[key]
            if isinstance(value, list):
                for i in range(len(value)):
                    new_set = copy.deepcopy(set)
                    new_set.__dict__[key] = value[i]
                    output_set.append(new_set)
                for_removal.append(set)
                break

    for set in for_removal:
        output_set.remove(set)
    
    print("Number of configs found for ", input_set.__class__ ,", : ", len(output_set))
    for set in output_set:
        yield set

wrapper_settings = TorchWrapperSettings(
                        normalize=[True, False], 
                        flatten_action=True,
                        skip_frames=[None, 1,2, 3])


hyperParams = HyperParams(
                        memory_size=10000, 
                        batch_size=[128,256], 
                        gamma=[0.95,0.99], 
                        tau=0.005,
                        eps_start=0.95,
                        eps_end=0.05,
                        eps_decay=4000,
                        lr=1e-4
                        )

options = GameOptions()
options.max_steps = [300,500]
options.max_projectiles_per_turret = 0
options.reward_type = [4,5]

hyperparameter_search(options, hyperParams, wrapper_settings)