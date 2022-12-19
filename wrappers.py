import torch
import gym
import numpy as np

class TorchWrapper(gym.Wrapper):
    
    """
    This class provides a wrapper for the environment to be used with pytorch:
    - Flattens and normalizes the observations to be between 0 and 1
    - Converts the observation to a torch tensor
    - Converts the action to a numpy array for the environment
    - Converts the reward to a torch tensor
    """
    def __init__(self, env, device="cpu", normalize=True):
        env = gym.wrappers.FlattenObservation(env)
        super(TorchWrapper, self).__init__(env)
        self.device = device
        self.normalize = normalize
        obs, _ = self.env.reset()
        self.obs_shape = obs.shape[0]

    def _observation(self, obs):
        if self.normalize:
            obs = obs.astype(np.float32)
            #normalize it to be between 0 and 1 using the max value defined in the observation space
            obs = obs/self.observation_space.high
        #convert to torch tensor
        obs = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(dim=0)
        return obs
    
    def reset(self):
        obs, info = self.env.reset()
        obs = self._observation(obs)
        return obs, info

    def step(self, action):
        if(len(action.shape) == 1):
            action = action.squeeze(dim=0).item()
        else:
            action = action.squeeze(dim=0).numpy()
        obs, reward, done, _, info = self.env.step(action)
        obs = self._observation(obs)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        return obs, reward, done, _, info

    def get_shapes(self):
        action_shape = (-1, len(self.env.action_space.nvec), np.max(self.env.action_space.nvec))
        num_actions_act = self.env.action_space.nvec[0]*self.env.action_space.nvec[1]
        return self.obs_shape, num_actions_act, action_shape