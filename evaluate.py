from game import Game, GameOptions
from settings import *
from wrappers import *
import imageio
from stable_baselines3 import PPO
import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

def create_env(options, settings, vectorized=False, monitor=False, render_mode=None, n_envs=6):
    
    if vectorized:
        env = SubprocVecEnv([lambda: GameWrapper(Game(options=options), wrapper_settings = settings) for i in range(n_envs)],
                            #start_method='spawn'
                            )
    else:
        env = Game(render_mode=render_mode, options=options)    
        env = GameWrapper(env, wrapper_settings = settings)
        
    if monitor:
        env = VecMonitor(env, info_keywords=("is_success",)) if vectorized else Monitor(env, info_keywords=("is_success",))
    return env

def get_max_min_reward(callbak: EvalCallback):
    """
    Returns the max and min reward in a callback of type EvalCallback
    """
    eval_rewards = callbak.evaluations_results
    if len(eval_rewards) == 0:
        print("No evaluation was done")
        return 0, 0

    eval_max_rewards = np.max(eval_rewards, axis=1)
    eval_min_rewards = np.min(eval_rewards, axis=1)

    print("Max rewards: ", eval_max_rewards)
    print("Min rewards: ", eval_min_rewards)

def get_best_length(callbak: EvalCallback):
    """
    Returns the best length and the corresponding reward in a
    callback of type EvalCallback
    """
    eval_lengths = callbak.evaluations_length
    if len(eval_lengths) == 0:
        print("No evaluation was done")
        return 0, 0, 0, 0

    eval_mean_lengths = np.mean(eval_lengths, axis=1)

    best_length_mean = np.min(eval_mean_lengths)
    best_length_index = np.argmin(eval_mean_lengths)
    best_length_std = np.std(eval_lengths, axis=1)[best_length_index]

    fastest_mean_reward = np.mean(callbak.evaluations_results, axis=1)[best_length_index]
    fastest_std_reward = np.std(callbak.evaluations_results, axis=1)[best_length_index]

    return best_length_mean, best_length_std, fastest_mean_reward, fastest_std_reward

def evaluate(model, options, settings, deterministic=True, n_episodes=5, save_gif_path=False, render=False, print_results=True, vectorized=False):
    assert not (save_gif_path and render), "Can't save gif and render at the same time" 
    images = []
    if render:
        render_mode = "human"
    elif save_gif_path:
        render_mode = "rgb_array"
    else:
        render_mode = None
        
    env = create_env(options, settings, vectorized=vectorized, monitor=False, render_mode=render_mode)

    obs, _ = env.reset()
    if render or save_gif_path:
        img = env.render()
    scores = []
    lenghts = []
    n_sucesses = 0
    for i in range(n_episodes):
        episode_score= 0
        n_steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if render or save_gif_path:
                if save_gif_path:
                    images.append(np.array(img))
                img = env.render()
            done = terminated or truncated
            episode_score += reward
            n_steps += 1
            if done:
                obs, _ = env.reset()
                scores.append(episode_score)
                lenghts.append(n_steps)
                if info["is_success"]:
                    n_sucesses += 1
                break
    if save_gif_path:
        path = save_gif_path + "game_capture_{}_{}.gif".format(str(deterministic),datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        imageio.mimsave(path, images, fps=60)
        print("Gif saved to {}".format(path), flush=True)

    if print_results:
        # Print number of episodes and boolean arguments to the function
        print("Evaluated {} episodes, deterministic={}. Results:".format(n_episodes, deterministic))
        print("  Average score: {:.2f} +/- {:.2f}".format(np.mean(scores), np.std(scores)))
        print("  Average length: {:.2f} +/- {:.2f}".format(np.mean(lenghts), np.std(lenghts)), flush=True)
        print("  Success rate: {:.2f}%".format(n_sucesses/n_episodes*100), flush=True)

    return np.mean(scores), np.std(scores), np.mean(lenghts), np.std(lenghts)


if __name__ == "__main__":
    options = GameOptions.from_yaml("configs/game_scenarios.yaml", "800x800-mid_barrier-no-proj")
    rewards = RewardScheme.from_yaml("configs/rewards.yaml", "config_3")
    settings = GameWrapperSettings(normalize=True, flatten_action=True, skip_frames=False, to_tensor=False)
    options.rew = rewards

    MODEL = "./opt-logs/ppo_midbarrier_config3/0/best_model.zip"
    model = PPO.load(MODEL)
    evaluate(model, options, settings, deterministic=True, n_episodes=10, save_gif_path=True, render=False)
    evaluate(model, options, settings, deterministic=False, n_episodes=10, save_gif_path=True, render=False)
    # evaluate(model, options, settings, deterministic=False, n_episodes=5, save_gif=False, render=True)
    evaluate(model, options, settings, deterministic=True, n_episodes=100, save_gif_path=False, render=True)
    evaluate(model, options, settings, deterministic=False, n_episodes=100, save_gif_path=False, render=True)

    

