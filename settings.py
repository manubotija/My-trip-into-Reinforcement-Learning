import yaml
from yaml import Loader, Dumper
import pygame


class BaseSettings():

    def __init__(self):
        pass

    def __str__(self):
        return str(self.get_dict())

    def get_dict(self):
        params =  self.__dict__
        # go over all the attributes and retrieve the dicts for those that BaseSettings
        for k,v in params.items():
            if isinstance(v, BaseSettings):
                params[k] = v.get_dict()
        #include the class name in the attributes
        params['class'] = self.__class__.__name__
        return params

    def _load_dict(params):
        # go over all the attributes and for those that are BaseSettings, create the object and call from_dict
        # the rest of the attributes are set directly
        for k,v in params.items():
            if isinstance(v, dict) and 'class' in v:
                class_name = v['class']
                if class_name in globals():
                    cls = globals()[class_name]
                    obj = cls.from_dict(v)
                    params[k] = obj
        #remove the class name from the dict
        if 'class' in params: 
            params.pop('class')
        return params
   
    @classmethod
    def from_dict(cls, params):
        params = cls._load_dict(params)
        obj = cls()
        obj.__dict__.update(params)
        return obj

    def to_yaml(self, path):
        yaml_str = yaml.dump(self.get_dict())
        with open(path, 'w') as f:
            f.write(yaml_str)
        return path

    def _load_yaml(path, section=None):
        with open(path, 'r') as f:
            yaml_str = f.read()
        params = yaml.load(yaml_str, Loader=Loader)
        if section is not None:
            params = params[section]
        return params
    
    @classmethod
    def from_yaml(cls, path, section=None):
        params = cls._load_yaml(path, section)
        obj = cls.from_dict(params)
        return obj
        

class Bounds(BaseSettings):
    def __init__(self, left=0, top=0, width=0, height=0):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def right(self):
        return self.left + self.width

class RewardScheme(BaseSettings):
    def __init__(self,
                    kill_penalty = -100,
                    win_reward = 500,
                    remaining_step_reward_factor = 1.,
                    collision_penalty = -1,
                    closer_reward_factor = 1.,
                    closer_reward_offset = 0,
                    further_penalty_factor = 1.,
                    further_penalty_offset = 0,
                    still_penalty = -1,
                ) -> None:
        self.kill_penalty = kill_penalty
        self.win_reward = win_reward
        self.remaining_step_reward_factor = remaining_step_reward_factor
        self.collision_penalty = collision_penalty
        self.closer_reward_factor = closer_reward_factor
        self.closer_reward_offset = closer_reward_offset
        self.further_penalty_factor = further_penalty_factor
        self.further_penalty_offset = further_penalty_offset
        self.still_penalty = still_penalty
    
    def normalize(self, max_value):
        self.kill_penalty /= max_value
        self.win_reward /= max_value
        self.remaining_step_reward_factor /= max_value
        self.collision_penalty /= max_value
        self.closer_reward_factor /= max_value
        self.closer_reward_offset /= max_value
        self.further_penalty_factor /= max_value
        self.further_penalty_offset /= max_value
        self.still_penalty /= max_value
        return self


class GameOptions(BaseSettings):
    
    def __init__(self, height = 800, 
                width=800, 
                n_turrets=3, 
                n_obstacles=3, 
                max_projectiles_per_turret=5, 
                fire_turret_step_delay=45,
                projectile_speed=5,
                turret_bounds : Bounds = None,
                obstacle_bounds : Bounds = None,
                gate_bounds : Bounds = None,
                player_bounds : Bounds = None,
                max_steps = 10000,
                instantiate_turrets = True,
                instantiate_obstacles = True,
                rew = RewardScheme(),
                touch_obstacle_kill = False,
                ) -> None:
        
        self.height = height
        self.width = width
        self.n_turrets = n_turrets
        self.n_obstacles = n_obstacles
        self.max_projectiles_per_turret = max_projectiles_per_turret
        self.fire_turret_step_delay = fire_turret_step_delay
        self.max_projectiles_per_turret = max_projectiles_per_turret
        self.fire_turret_step_delay = fire_turret_step_delay
        self.projectile_speed = projectile_speed
        self.max_steps = max_steps
        self.instantiate_turrets = instantiate_turrets
        self.instantiate_obstacles = instantiate_obstacles
        self.rew = rew
        self.touch_obstacle_kill = touch_obstacle_kill
        
        if turret_bounds == None:
            self.turret_bounds = Bounds(0, 0, width*0.95, height*0.95)
        else:
            self.turret_bounds = turret_bounds
        if obstacle_bounds == None:
            self.obstacle_bounds = Bounds(0, 0, width*0.95, height*0.95)
        else:
            self.obstacle_bounds = obstacle_bounds
        if gate_bounds == None:
            self.gate_bounds = Bounds(0, 0, width*0.95, height*0.95)
        else:
            self.gate_bounds = gate_bounds
        if player_bounds == None:
            self.player_bounds = Bounds(0, 0, width*0.95, height*0.95)
        else: 
            self.player_bounds = player_bounds

