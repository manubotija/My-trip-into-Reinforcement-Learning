import torch
import gym
import numpy as np


class GameWrapperSettings():
    def __init__(self, normalize=True, flatten_action=True, skip_frames=None, to_tensor=None):
        self.normalize = normalize
        self.flatten_action = flatten_action
        self.skip_frames = skip_frames
        self.to_tensor = to_tensor
    def __str__(self) -> str:
        return str(self.get_params())
    def get_params(self):
        return self.__dict__
    def from_params(params):
        settings = GameWrapperSettings()
        settings.__dict__.update(params)
        return settings
    
class GameWrapper(gym.Wrapper):
    """
    This class provides a wrapper for the environment with multiple (optional) functions:
    - Flattens the observation
    - Normalize the observations to be between -1 and 1
    - Convert the observation to a torch tensor
    - Convert the action from tensor to a numpy array for the environment
    - Convert the reward to a torch tensor
    - Skip frames
    - Insert a time limit
    """
    def __init__(self, env, device="cpu", wrapper_settings=GameWrapperSettings()):
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env.options.max_steps)
        super(GameWrapper, self).__init__(env)
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
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.observation_space.shape, dtype=np.float32)

    def _observation(self, obs):
        if self.normalize:
            obs = obs.astype(np.float32)
            #normalize it to be between -1 and 1 using the max value defined in the observation space
            obs = (obs/(self.original_observation_space.high/2) - 1).astype(np.float32)
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
            
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.skip_frames is not None:
            for _ in range(self.skip_frames):
                if terminated or truncated:
                    break
                #repeat action for all skipped frames
                obs, r, terminated, truncated, info = self.env.step(action)
                reward+=r
        obs = self._observation(obs)
        if self.to_tensor:
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        return obs, reward, terminated, truncated, info