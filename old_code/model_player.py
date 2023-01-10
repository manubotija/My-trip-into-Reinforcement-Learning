import torch
from itertools import count
from game import Game, GameOptions
from wrappers import GameWrapper, GameWrapperSettings
import numpy as np
from old_code.models import *
from torchinfo import summary
from old_code.utils import load_model, plot_model_training_history
from pygame import Rect
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Class that takes a model and a game and plays the game using the model.
"""
class ModelPlayer:
    def __init__(self, model, game, device, action_shape,save_video=False):
        self.model = model
        self.game = game
        self.device=device
        self.action_shape = action_shape
        self._save_video = False
        if self.game.render_mode == "rgb_array" and save_video:
            self._init_video()
            self._save_video = True

    def play(self, num_episodes=1):
        obs, info = self.game.reset()
        successful_count = 0
        step_counter = []
        total_score = 0
        for i_episode in range(num_episodes):
            for t in count():
                #input = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(dim=0)
                output = self.model(obs).view(self.action_shape)
                action = torch.max(output, dim=-1).indices
                obs, reward, done, _, info = self.game.step(action)
                frame = self.game.render()
                if self.game.render_mode == "rgb_array" and self._save_video:
                    self._save_frame(frame)
                if done:
                    print("Episode: {}/{} Duration: {}. Score: {}".format(i_episode, num_episodes, t + 1, info["score"]), end="\r", flush=True)
                    if reward >= 100:
                        successful_count += 1
                    step_counter.append(t+1)
                    self.game.reset()
                    break
        print("Successful episodes: {}/{} ({}%)".format(successful_count, num_episodes, successful_count/num_episodes*100), "Average steps per episode", np.mean(step_counter))
        if self.game.render_mode == "rgb_array" and self._save_video:
            self.save_video()

    def _init_video(self):
        self.video = []
        self.video.append(self.game.render())
    def _save_frame(self, frame):
        self.video.append(frame)
    #creates an mp4 out of the numpy arrays stored in self.video
    def save_video(self, path="video_{}.mp4".format(datetime.now().strftime("%Y%m%d-%H%M%S"))):
        import imageio
        imageio.mimwrite(path, self.video, fps=60)
        


# MODEL_PATH = "checkpoints/20221223-113034"
MODEL_PATH = "checkpoints/20221229-180150"

plot_model_training_history(MODEL_PATH)
policy_model, target_model, options, wrapper_settings = load_model(MODEL_PATH)
options = GameOptions.from_params(options)



env = GameWrapper(Game(render_mode="human", options=options, render_fps=120), wrapper_settings=GameWrapperSettings.from_params(wrapper_settings), device=device)
# model_class is a str get actual class
obs_shape, num_actions_act, action_shape = env.get_shapes()

#summary(policy_model, input_size=(1, obs_shape), device=device.type)

player = ModelPlayer(policy_model, env, device, action_shape, save_video=False)
player.play(num_episodes=100)

player = ModelPlayer(target_model, env, device, action_shape)
player.play(num_episodes=100)