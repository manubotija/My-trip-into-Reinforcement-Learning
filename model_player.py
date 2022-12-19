import torch
from itertools import count
from game import Game, GameOptions
from wrappers import TorchWrapper
import numpy as np
from torchinfo import summary
from utils import load_model, plot_model_training_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Class that takes a model and a game and plays the game using the model.
"""
class ModelPlayer:
    def __init__(self, model, game, device):
        self.model = model
        self.game = game
        self.device=device

    def play(self, num_episodes=1):
        obs, info = self.game.reset()
        successful_count = 0
        step_counter = []
        for i_episode in range(num_episodes):
            for t in count():
                input = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(dim=0)
                output = self.model(input)
                action = torch.max(output, dim=-1).indices
                obs, reward, done, _, info = self.game.step(action.squeeze(dim=0).numpy())
                self.game.render()
                if done:
                    print("Episode: {}/{} Duration: {}. Final reward: {}".format(i_episode, num_episodes, t + 1, reward), end="\r", flush=True)
                    if reward > 0:
                        successful_count += 1
                    step_counter.append(t+1)
                    self.game.reset()
                    break
        print("Successful episodes: {}/{} ({}%)".format(successful_count, num_episodes, successful_count/num_episodes*100), "Average steps per episode", np.mean(step_counter))



model_path = "20221211-225256"

#plot_model_training_history(model_path)
model_class, weights, options = load_model(model_path)
options = GameOptions.from_params(options)
env = TorchWrapper(Game(render_mode=None, options=options, render_fps=200))
# model_class is a str get actual class
model_class = globals()[model_class]


model = model_class(env.observation_space.shape[0], np.sum(env.action_space.nvec), (-1, len(env.action_space.nvec), np.max(env.action_space.nvec))).to(device)
model.load_state_dict(weights)
summary(model, input_size=(1, env.observation_space.shape[0]), device=device.type)
player = ModelPlayer(model, env, device)
player.play(num_episodes=1000)