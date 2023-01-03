
from typing import Any, Dict

import numpy as np
import optuna
from optuna.trial import TrialState
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn
from game import Game, GameOptions
from wrappers import *
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import pprint
from evaluate import evaluate, create_env


def _sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = "constant"
    
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
def sample_game_params(trial: optuna.Trial) -> Dict[str, Any]:
    pass


def objective(trial, options, settings, timesteps=600_000, deterministic_eval=False):

    sampled_hyperparams = _sample_ppo_params(trial)
    
    env = create_env(options, settings, vectorized=True, monitor=True, render_mode=None, n_envs=6)
    eval_env = create_env(options, settings, vectorized=False, monitor=True, render_mode=None)
    
    study_name = trial.study.study_name
    opt_path = f"./opt-logs/{study_name}/{str(trial.number)}/"
    tb_path = f"./opt-tb/{study_name}/"

    model = PPO("MlpPolicy", 
                env, 
                seed=None, 
                verbose=0, 
                tensorboard_log=tb_path,
                **sampled_hyperparams)
   
    eval_callback = EvalCallback(eval_env, 
                                best_model_save_path=opt_path,
                                log_path=opt_path, 
                                eval_freq=100_000, 
                                n_eval_episodes=500,
                                deterministic=deterministic_eval, 
                                render=False)

    try:
        model.learn(total_timesteps=timesteps, progress_bar=False, callback=eval_callback, tb_log_name=str(trial.number))
        model.env.close()
        eval_env.close()
    except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()
    
    del model.env, eval_env
    del model

    return eval_callback.best_mean_reward
    
if __name__ == "__main__":

    options = GameOptions()
    options.max_steps = 300
    options.max_projectiles_per_turret = 0
    options.instantiate_obstacles = True
    options.instantiate_turrets = True
    options.reward_type = 4

    settings = GameWrapperSettings(normalize=True, flatten_action=True, skip_frames=None, to_tensor=False)

    #create optuna study
    study_name="ppo_opt"
    study = optuna.create_study(study_name=study_name, storage="sqlite:///db2.sqlite3", direction="maximize", load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, options, settings),
        n_trials=2,
        n_jobs=1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  ID: ", trial.number)
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    #Load model from best trial
    model = PPO.load(f"./opt-logs/{study_name}/{trial.number}/best_model.zip")
    evaluate(model, options, settings, deterministic=True, n_episodes=5, save_gif=True, render=False)
    evaluate(model, options, settings, deterministic=False, n_episodes=5, save_gif=True, render=False)
    evaluate(model, options, settings, deterministic=True, n_episodes=1000, save_gif=False, render=False)
    evaluate(model, options, settings, deterministic=False, n_episodes=1000, save_gif=False, render=False)