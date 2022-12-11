
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class SimpleLinearModel(nn.Module):
    def __init__(self, obs_dim, act_dim, output_shape):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, act_dim)
        self.output_shape = output_shape
        

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(self.output_shape) 

