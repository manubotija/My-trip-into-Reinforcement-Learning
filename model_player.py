import torch
from itertools import count
from models import SimpleLinearModel
from gym.wrappers import FlattenObservation
from game import Game, GameOptions
import numpy as np
from torchinfo import summary

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
        for i_episode in range(num_episodes):
            for t in count():
                input = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(dim=0)
                output = self.model(input)
                action = torch.max(output, dim=-1).indices
                obs, reward, done, _, info = self.game.step(action.squeeze(dim=0).numpy())
                self.game.render()
                if done:
                    print("Episode: {}/{} Duration: {}. Final reward: {}".format(i_episode, num_episodes, t + 1, reward))
                    self.game.reset()
                    break




options = GameOptions(  height=600, 
                        width=400, 
                        n_obstacles=1, 
                        n_turrets=1, 
                        max_projectiles_per_turret=2, 
                        fire_turret_step_delay=100
                        )
env = FlattenObservation(Game(render_mode="human", options=options, render_fps=60)) 
obs, _ = env.reset()
action_shape = (-1, len(env.action_space.nvec), np.max(env.action_space.nvec))

checkpoint_path = "checkpoints/model_600x600_1t_2p_1o_100d_1670189714.664002.pt"
model = SimpleLinearModel(obs.shape[0], np.sum(env.action_space.nvec), action_shape).to(device)
model.load_state_dict(torch.load(checkpoint_path))
summary(model, input_size=(1, obs.shape[0]), device=device.type)

player = ModelPlayer(model, env, device)
player.play(num_episodes=20)