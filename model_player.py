import torch
from itertools import count
from game import Game, GameOptions
from wrappers import TorchWrapper, TorchWrapperSettings
import numpy as np
from models import *
from torchinfo import summary
from utils import load_model, plot_model_training_history
from pygame import Rect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Class that takes a model and a game and plays the game using the model.
"""
class ModelPlayer:
    def __init__(self, model, game, device, action_shape):
        self.model = model
        self.game = game
        self.device=device
        self.action_shape = action_shape

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
                self.game.render()
                if done:
                    print("Episode: {}/{} Duration: {}. Score: {}".format(i_episode, num_episodes, t + 1, info["score"]), end="\r", flush=True)
                    if reward > 0:
                        successful_count += 1
                    step_counter.append(t+1)
                    self.game.reset()
                    break
        print("Successful episodes: {}/{} ({}%)".format(successful_count, num_episodes, successful_count/num_episodes*100), "Average steps per episode", np.mean(step_counter))



model_path = "checkpoints/20221219-222555"

#plot_model_training_history(model_path)
policy_weights, target_weights, policy_model_class, target_model_class, options, wrapper_settings = load_model(model_path)
options = GameOptions.from_params(options)

# options.width = 800
# options.height = 600
# options.player_bounds = Rect(0, options.height*0.8, options.width, options.height*0.2)
# options.gate_bounds= Rect(0, 0, options.width, options.height*0.2)

env = TorchWrapper(Game(render_mode="human", options=options, render_fps=200), wrapper_settings=TorchWrapperSettings.from_params(wrapper_settings), device=device)
# model_class is a str get actual class
model_class = globals()[policy_model_class]

obs_shape, num_actions_act, action_shape = env.get_shapes()
policy_model = model_class(obs_shape, num_actions_act).to(device)

#model = model_class(env.observation_space.shape[0], np.sum(env.action_space.nvec), (-1, len(env.action_space.nvec), np.max(env.action_space.nvec))).to(device)
policy_model.load_state_dict(policy_weights)
summary(policy_model, input_size=(1, env.observation_space.shape[0]), device=device.type)
player = ModelPlayer(policy_model, env, device, action_shape)
player.play(num_episodes=100)