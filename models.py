
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class ModelParams():
    def __init__(self, obs_dim, act_dim, hidden_dim=128, hidden_layers=2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
    def get_params(self):
        return {
            "obs_dim": self.obs_dim, 
            "act_dim": self.act_dim, 
            "hidden_dim": self.hidden_dim, 
            "hidden_layers": self.hidden_layers}
    def from_params(params):
        return ModelParams(
                params["obs_dim"], 
                params["act_dim"], 
                params["hidden_dim"], 
                params["hidden_layers"])
    def __str__(self):
        return str(self.get_params())

class SimpleLinearModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.params = model_params
        self.obs_dim = model_params.obs_dim
        self.act_dim = model_params.act_dim
        self.fc1 = nn.Linear(model_params.obs_dim, model_params.hidden_dim)
        for i in range(model_params.hidden_layers):
            setattr(self, "fc"+str(i+2), nn.Linear(model_params.hidden_dim, model_params.hidden_dim))
        self.fc_final = nn.Linear(model_params.hidden_dim, model_params.act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        for i in range(self.params.hidden_layers):
            x = F.relu(getattr(self, "fc"+str(i+2))(x))
        x = self.fc_final(x)
        return x

