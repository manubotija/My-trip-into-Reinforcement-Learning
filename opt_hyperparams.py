
from typing import Any, Dict

import numpy as np
import optuna
from optuna.trial import TrialState
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn
from game import Game, GameOptions
from settings import *
from wrappers import *
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import pprint
from _evaluate import _evaluate, create_env, get_best_length
from callbacks import TrialEvalCallback
import yaml

def _sample_game_rewards(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for game rewards.
    :param trial:
    :return: dict with reward settings for each parameter of RewardScheme
    """
    # kill_penalty = trial.suggest_int("kill_penalty", -1000, -100, step=100)
    win_reward = trial.suggest_int("win_reward", 0, 1000, step=100)
    remaining_step_reward_factor = trial.suggest_float("remaining_step_reward_factor", 0, 3.0, step=0.1)
    collision_penalty = trial.suggest_int("collision_penalty", -10.0, 0.0, step=1)
    closer_reward_factor = trial.suggest_float("closer_reward_factor", 0.0, 3.0, step=0.1)
    closer_reward_offset = trial.suggest_int("closer_reward_offset", 0, 10, step=1)
    further_penalty_factor = trial.suggest_float("further_penalty_factor", 0.0, 3.0, step=0.1)
    further_penalty_offset = trial.suggest_int("further_penalty_offset", -10, 0, step=1)
    still_penalty = trial.suggest_int("still_penalty", -10, 0, step=1)
    
    return {
        # "kill_penalty": kill_penalty,
        "win_reward": win_reward,
        "remaining_step_reward_factor": remaining_step_reward_factor,
        "collision_penalty": collision_penalty,
        "closer_reward_factor": closer_reward_factor,
        "closer_reward_offset": closer_reward_offset,
        "further_penalty_factor": further_penalty_factor,
        "further_penalty_offset": further_penalty_offset,
        "still_penalty": still_penalty
    }

    
def _sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lr_schedule = "constant"
    
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["medium", "big"])
    
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
    # if batch_size > n_steps:
    #     batch_size = n_steps

    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "big": [dict(pi=[256, 256, 256], vf=[256, 256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        # "batch_size": batch_size,
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


def find_best_model_in_study(study, options, settings, deterministic_eval=True, vectorized_env=False):
    """Goes over all the models saved in opt-logs during the study and evaluates them"""
    base_path = f"./opt-logs/{study.study_name}/"
    completed_trial_numbers = [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_score = -np.inf
    best_score_std = 0
    best_score_lenght = 0
    best_score_length_std = 0
    shortest_length = np.inf
    shortest_length_std = 0
    shortest_score = -np.inf
    shortest_score_std = 0
    best_trial_by_score = None
    best_trial_by_length = None
    for trial_number in completed_trial_numbers:
        path = base_path + str(trial_number) + "/best_model.zip"
        model = PPO.load(path)
        mean_score, std_score, mean_length, std_length = _evaluate(model, options, settings, deterministic=deterministic_eval, n_episodes=1000, save_gif_path=False, render=False, print_results=False, vectorized=vectorized_env)
        if mean_score > best_score:
            best_score = mean_score
            best_score_std = std_score
            best_score_lenght = mean_length
            best_score_length_std = std_length
            best_trial_by_score = trial_number
        if mean_length < shortest_length:
            shortest_length = mean_length
            shortest_length_std = std_length
            shortest_score = mean_score
            shortest_score_std = std_score
            best_trial_by_length = trial_number
        
        print(f"Best trial by score: score: {best_score} +/- {best_score_std} - Best trial by length: {shortest_length} +/- {shortest_length_std} Trials left: {len(completed_trial_numbers) - completed_trial_numbers.index(trial_number)}     ", end="\r")

    print(f"Best trial by score: {best_trial_by_score}", f"  score: {best_score} +/- {best_score_std}", f"  length: {best_score_lenght} +/- {best_score_length_std}", sep="\n")
    print(f"Best trial by length: {best_trial_by_length}", f"  score: {shortest_score} +/- {shortest_score_std}", f"  length: {shortest_length} +/- {shortest_length_std}", sep="\n")

    return best_trial_by_score, best_trial_by_length

def objective(
    trial, 
    options: GameOptions, 
    settings, 
    timesteps=600_000, 
    deterministic_eval=False, 
    optimize_for_length=False,
    optimize_hyperparams=True,
    optimize_rewards=True,):

    if optimize_hyperparams:
        sampled_hyperparams = _sample_ppo_params(trial)
    else:
        sampled_hyperparams = {}
    
    if optimize_rewards:
        reward_params = _sample_game_rewards(trial)
        options.rew = RewardScheme.from_dict(reward_params)
    
    n_envs = 6
    env = create_env(options, settings, vectorized=True, monitor=True, render_mode=None, n_envs=n_envs)
    eval_env = create_env(options, settings, vectorized=False, monitor=True, render_mode=None)
    
    study_name = trial.study.study_name
    opt_path = f"./opt-logs/{study_name}/{str(trial.number)}/"
    tb_path = f"./opt-tb/{study_name}/"

    model = PPO("MlpPolicy", 
                env, 
                seed=None, 
                verbose=0, 
                tensorboard_log=tb_path,
                **sampled_hyperparams
                )

    eval_freq = timesteps // (10*n_envs)
   
    eval_callback = TrialEvalCallback(
                                eval_env = eval_env,
                                trial=trial,
                                best_model_save_path=opt_path,
                                log_path=opt_path, 
                                eval_freq=eval_freq, 
                                n_eval_episodes=200,
                                deterministic=deterministic_eval, 
                                optimize_for_length=optimize_for_length)

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
            if sampled_hyperparams:
                print(e)
                print("============")
                print("Sampled hyperparams:")
                pprint(sampled_hyperparams)
            if reward_params:
                print("============")
                print("Sampled reward params:")
                pprint(reward_params)
            raise optuna.exceptions.TrialPruned()
    

    is_pruned = eval_callback.is_pruned

    del model.env, eval_env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    if optimize_for_length:
        best_length, _, _, _ = get_best_length(eval_callback)
        return best_length
    else:
        return eval_callback.best_mean_reward


   
if __name__ == "__main__":

    options = GameOptions.from_yaml("configs/game_scenarios.yaml", "800x800-mid_barrier-no-proj")
    rewards = RewardScheme.from_yaml("configs/rewards.yaml", "config_3")
    options.rew = rewards.normalize(rewards.win_reward)
    settings = GameWrapperSettings(normalize=True, flatten_action=True, skip_frames=4, to_tensor=False)

    #create optuna study
    STUDY_NAME="ppo_midbarrier_config3"
    
    study = optuna.create_study(
        study_name=STUDY_NAME, 
        storage="sqlite:///db.sqlite3", 
        direction="maximize", 
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    )
    
    # find_best_model_in_study(study, options, settings, deterministic_eval=True)
    # quit()

    study.optimize(
        lambda trial: objective(
                                trial, 
                                options, 
                                settings,
                                timesteps=1_000_000,
                                deterministic_eval=False,
                                optimize_for_length=False,
                                optimize_hyperparams=True,
                                optimize_rewards=False),
        n_trials=300,
        n_jobs=1,
        timeout=8*60*60,
        show_progress_bar=False)

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
    model = PPO.load(f"./opt-logs/{STUDY_NAME}/{trial.number}/best_model.zip")
    _evaluate(model, options, settings, deterministic=True, n_episodes=10, save_gif_path=True, render=False)
    _evaluate(model, options, settings, deterministic=False, n_episodes=10, save_gif_path=True, render=False)
    _evaluate(model, options, settings, deterministic=True, n_episodes=1000, save_gif_path=False, render=False)
    _evaluate(model, options, settings, deterministic=False, n_episodes=1000, save_gif_path=False, render=False)

    #save params to yaml file
    with open(f"./opt-logs/{STUDY_NAME}/best_trial.yaml", "w") as f:
        f.write(yaml.dump(trial.params))