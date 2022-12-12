
from models import *
from game import Game, GameOptions, NormalizedFLattenedObservation
from gym.wrappers import FlattenObservation
from trainer import Trainer, HyperParams
import numpy as np
import matplotlib.pyplot as plt
import time
from pygame import Rect
from utils import *
import sys

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(options, model_class, hyperparams=None, reward_type=1, model_weights=None):
    env = NormalizedFLattenedObservation(Game(  render_mode=None, 
                                                options=options, 
                                                render_fps=1000,
                                                reward_type=reward_type,
                                                )) 
    obs, _ = env.reset()
    action_shape = (-1, len(env.action_space.nvec), np.max(env.action_space.nvec))

    # target_model is the Q network we are training
    target_model = model_class(obs.shape[0], np.sum(env.action_space.nvec), action_shape).to(device)
    # aux_model helps us do soft updates
    aux_model = model_class(obs.shape[0], np.sum(env.action_space.nvec), action_shape).to(device)

    if model_weights is not None:
        target_model.load_state_dict(model_weights)
        aux_model.load_state_dict(model_weights)

    optimizer = optim.RMSprop(target_model.parameters())

    trainer = Trainer(target_model=target_model, 
                        aux_model=aux_model, 
                        env=env, 
                        optimizer=optimizer, 
                        action_shape=action_shape, 
                        device=device,
                        hyperparams=hyperparams)
    
    durations, rewards, success, loss, score, speed = trainer.do_train(
                                                        num_episodes = 200, 
                                                        target_max_steps=100, 
                                                        target_success_window=20)
    
    weights_path = ""

    if success:
        weights_path = save_model(  target_model,
                                    trainer=trainer,
                                    options=options,
                                    rewards=rewards,
                                    durations=durations,
                                    loss=loss)

    return weights_path, success, score, speed


def hyperparameter_search(options, hyperParams):
# Hyperparameter search
    gamma_list = [0.95, 0.99, 0.9, 0.999]
    eps_decay_list = [4000]
    eps_end_list = [0.1, 0.05]
    batch_size_list = [200]
    reward_type_list = [2,3,4,1]

    max_tries_per_config = 10

    #iterate over all configs
    best_score = 0
    best_speed = sys.maxsize
    best_hyperparams = None
    for batch_size in batch_size_list:
        for gamma in gamma_list:
            for eps_decay in eps_decay_list:
                for eps_end in eps_end_list:
                    for reward_type in reward_type_list:
                        hyperParams.gamma = gamma
                        hyperParams.eps_decay = eps_decay
                        hyperParams.eps_end = eps_end
                        hyperParams.batch_size = batch_size
                        print("\n=====Trying config: ", hyperParams, "======\n")
                        for i in range(max_tries_per_config):
                            weights_path, success, score, speed = run_training(
                                                                        options, 
                                                                        SimpleLinearModel, 
                                                                        hyperParams, 
                                                                        reward_type)
                            if speed < best_speed:
                                best_score = score
                                best_speed = speed
                                best_hyperparams = hyperParams
                                print("=====New best speed ", speed, "with score", score, "======")
                            if success: 
                                print("=====Success! Config: ", hyperParams, "======")
                                return weights_path

    print("Failed to solve task with any config. Best config: ", best_hyperparams, " with speed: ",best_speed, " and score: ", best_score, "======")
    return None



hyperParams = HyperParams(
                        memory_size=5000, 
                        batch_size=100, 
                        gamma=0.89, 
                        param_update_rate=0.01,
                        eps_start=0.95,
                        eps_end=0.05,
                        eps_decay=4000
                        )

options = GameOptions(  height=300, 
                        width=300, 
                        n_obstacles=0, 
                        n_turrets=0, 
                        max_projectiles_per_turret=0, 
                        fire_turret_step_delay=0,
                        max_steps=1000
                        )

options.player_bounds = Rect(0, 0, options.width, options.height)
options.gate_bounds= Rect(0, 0, options.width, options.height)

weights_path_1 = hyperparameter_search(options, hyperParams)
quit()
#weights_path, success, score = run_training(options, SimpleLinearModel, hyperParams)

model_path = "20221211-225256"

#plot_model_training_history(model_path)
_ , weights, _ = load_model(model_path)
weights_path, success, score = run_training(options, SimpleLinearModel, hyperParams, weights)

quit()

options.fire_turret_step_delay = 100
