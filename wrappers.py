import torch
import gym
import numpy as np


class TorchWrapperSettings():
    def __init__(self, normalize=False, flatten_action=False, skip_frames=None, to_tensor=True):
        self.normalize = normalize
        self.flatten_action = flatten_action
        self.skip_frames = skip_frames
        self.to_tensor = to_tensor
    def __str__(self) -> str:
        return str(self.get_params())
    def get_params(self):
        return self.__dict__
    def from_params(params):
        settings = TorchWrapperSettings()
        settings.__dict__.update(params)
        return settings
    
class TorchWrapper(gym.Wrapper):
    """
    This class provides a wrapper for the environment to be used with pytorch:
    - Flattens and normalizes the observations to be between 0 and 1
    - Converts the observation to a torch tensor
    - Converts the action to a numpy array for the environment
    - Converts the reward to a torch tensor
    """
    def __init__(self, env, device="cpu", wrapper_settings=TorchWrapperSettings()):
        env = gym.wrappers.FlattenObservation(env)
        super(TorchWrapper, self).__init__(env)
        self.device = device
        self.normalize = wrapper_settings.normalize
        self.flatten_action = wrapper_settings.flatten_action
        self.skip_frames = wrapper_settings.skip_frames
        self.to_tensor = wrapper_settings.to_tensor

        obs, _ = self.env.reset()
        self.obs_shape = obs.shape[0]
        
        #if action space is multidiscrete, convert it to a single discrete space with as many combinations as possible
        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete) and self.flatten_action:
            self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1]**self.env.action_space.nvec[0])

        if self.normalize:
            self.original_observation_space = self.observation_space
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_space.shape, dtype=np.float32)

    def _observation(self, obs):
        if self.normalize:
            obs = obs.astype(np.float32)
            #normalize it to be between 0 and 1 using the max value defined in the observation space
            obs = (obs/self.original_observation_space.high).astype(np.float32)
        if self.to_tensor:
            #convert to torch tensor
            obs = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(dim=0)
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._observation(obs)
        return obs, info

    def step(self, action):
        
        if self.flatten_action:
            shape = [self.env.action_space.nvec[1] for _ in range(self.env.action_space.nvec[0])]
            if self.to_tensor:
                action = action.cpu().numpy()
            action = np.unravel_index(action, shape)
        elif(len(action.shape) == 1):
            action = action.squeeze(dim=0)
            if self.to_tensor:
                action = action.cpu().item()
        else:
            action = action.squeeze(dim=0)
            if self.to_tensor:
                action = action.cpu().numpy()
            
        obs, reward, done, _, info = self.env.step(action)

        if self.skip_frames is not None:
            for _ in range(self.skip_frames):
                if done:
                    break
                #action NOOP
                action = self.env.get_noop_actions()
                obs, r, done, _, info = self.env.step(action)
                reward+=r
        obs = self._observation(obs)
        if self.to_tensor:
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        return obs, reward, done, _, info

    def get_shapes(self):
        if not self.flatten_action:
            action_shape = (-1, len(self.env.action_space.nvec), np.max(self.env.action_space.nvec))
            num_actions_act = self.env.action_space.nvec[0]*self.env.action_space.nvec[1]
        else:
            action_shape = (-1, self.action_space.n,)
            num_actions_act = self.action_space.n

        return self.obs_shape, num_actions_act, action_shape
