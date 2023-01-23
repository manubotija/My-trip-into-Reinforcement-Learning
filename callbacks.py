import optuna
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional
import numpy as np

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        optimize_for_length: bool = False,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.optimize_for_length = optimize_for_length

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            if self.optimize_for_length:
                last_mean_length = np.mean(self.evaluations_length[-1, :])
                self.trial.report(last_mean_length, self.eval_idx) 
            else:
                self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            self.eval_idx += 1
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True