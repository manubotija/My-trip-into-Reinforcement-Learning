import torch
import gym
import numpy as np
import cv2

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
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._observation(obs)
        return obs, info

    def step(self, action):
        
        if self.flatten_action:
            shape = [self.env.action_space.nvec[1] for _ in range(self.env.action_space.nvec[0])]
            action = np.unravel_index(action, shape)
            
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.skip_frames is not None:
            for _ in range(self.skip_frames):
                if terminated or truncated:
                    break
                #repeat action for all skipped frames
                obs, r, terminated, truncated, info = self.env.step(action)
                reward+=r
        obs = self._observation(obs)
        return obs, reward, terminated, truncated, info


class ImageWrapper(gym.Wrapper):
    """ wraps a game environment in order to:
    1. resize the observation by a given factor
    2. optionally compute the difference between two consecutive frames (TODO)
    3. covert the image to black and white
    """

    def __init__(self, env, factor: int = 16, diff: bool = False):

        assert env.render_mode == "rgb_array", "ImageWrapper only works with rgb_array render mode"

        super(ImageWrapper, self).__init__(env)
        self.factor = factor
        self.diff = diff
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs.shape[0], obs.shape[1], 1), dtype=np.uint8)

    def _observation(self):
        obs = self.env.render()
        # convert the image to greyscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # resize the observation by a given factor
        obs = cv2.resize(obs, (obs.shape[1] // self.factor, obs.shape[0] // self.factor), interpolation=cv2.INTER_AREA)
        # add a channel dimension
        obs = np.expand_dims(obs, axis=2)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._observation()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._observation()
        return obs, reward, terminated, truncated, info
