from game import Game, GameOptions
from wrappers import *
import imageio
from stable_baselines3 import PPO
import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

def create_env(options, settings, vectorized=False, monitor=False, render_mode=None, n_envs=6):
    
    if vectorized:
        env = SubprocVecEnv([lambda: GameWrapper(Game(options=options), wrapper_settings = settings) for i in range(n_envs)])
    else:
        env = Game(render_mode=render_mode, options=options)    
        env = GameWrapper(env, wrapper_settings = settings)
        
    if monitor:
        env = VecMonitor(env) if vectorized else Monitor(env)
    return env

def evaluate(model, options, settings, deterministic=True, n_episodes=5, save_gif=False, render=False):
    assert not (save_gif and render), "Can't save gif and render at the same time" 
    images = []
    if render:
        render_mode = "human"
    elif save_gif:
        render_mode = "rgb_array"
    else:
        render_mode = None
        
    env = create_env(options, settings, vectorized=False, monitor=False, render_mode=render_mode)

    obs, _ = env.reset()
    if render or save_gif:
        img = env.render()
    scores = []
    lenghts = []
    for i in range(n_episodes):
        episode_score= 0
        n_steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if render or save_gif:
                if save_gif:
                    images.append(np.array(img))
                img = env.render()
            done = terminated or truncated
            episode_score += reward
            n_steps += 1
            if done:
                obs, _ = env.reset()
                scores.append(episode_score)
                lenghts.append(n_steps)
                break
    if save_gif:
        path = "./temp/game_capture_{}_{}.gif".format(str(deterministic),datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        imageio.mimsave(path, images, fps=60)
        print("Gif saved to {}".format(path))

    # Print number of episodes and boolean arguments to the function
    print("Evaluated {} episodes, deterministic={}. Results:".format(n_episodes, deterministic))
    print("  Average score: {:.2f} +/- {:.2f}".format(np.mean(scores), np.std(scores)))
    print("  Average length: {:.2f} +/- {:.2f}".format(np.mean(lenghts), np.std(lenghts)))

    

if __name__ == "__main__":
    options = GameOptions()
    options.max_steps = 300
    options.max_projectiles_per_turret = 0
    options.instantiate_obstacles = True
    options.instantiate_turrets = True
    options.reward_type = 4
    settings = GameWrapperSettings(normalize=True, flatten_action=True, skip_frames=None, to_tensor=False)

    MODEL = "eval_logs/PP0_new_rewards_inst/best_model.zip"
    model = PPO.load(MODEL)
    evaluate(model, options, settings, deterministic=True, n_episodes=10, save_gif=True, render=False)
    evaluate(model, options, settings, deterministic=False, n_episodes=10, save_gif=True, render=False)
    # evaluate(model, options, settings, deterministic=False, n_episodes=5, save_gif=False, render=True)
    evaluate(model, options, settings, deterministic=True, n_episodes=100, save_gif=False, render=False)
    evaluate(model, options, settings, deterministic=False, n_episodes=100, save_gif=False, render=False)

    

