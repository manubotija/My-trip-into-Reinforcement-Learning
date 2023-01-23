# Import and initialize the pygame library
import pygame
import gym
import math
from gym import spaces
import numpy as np
from sprites import *
from settings import GameOptions
import time
# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
   
class Game(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], "default_render_fps": 60}
    FIRE_TURRET = pygame.USEREVENT + 1

    def __init__(self, render_mode=None, render_fps=None, options=None) -> None:

        pygame.init()

        if options is None:
            self.options = GameOptions()
        else:
            self.options = options

        # Set up the drawing window
        if render_mode == "human":
            self.screen = pygame.display.set_mode([self.options.width, self.options.height])
            self.font = pygame.font.SysFont(None, 24)

        self.action_space = spaces.MultiDiscrete([2, 3])

        # define observation space
        N_COORD = 4
        
        W = self.options.width
        H = self.options.height

        single_obs_size = N_COORD
        single_high_obs = np.array([W,H] * (N_COORD//2))
        single_low_obs = np.array([0] * (N_COORD))

        turrets_obs_size = self.options.n_turrets * N_COORD
        turrets_high_obs = np.array([W,H] * (turrets_obs_size//2))
        turrets_low_obs = np.array([0] * (turrets_obs_size))

        obstacles_obs_size = self.options.n_obstacles * N_COORD
        obstacles_high_obs = np.array([W,H] * (obstacles_obs_size//2))
        obstacles_low_obs = np.array([0] * (obstacles_obs_size))

        proj_obs_size = self.options.n_turrets * self.options.max_projectiles_per_turret * N_COORD
        proj_high_obs = np.array([W,H] * (proj_obs_size//2))
        proj_low_obs = np.array([0] * (proj_obs_size))

        # create observation space as a dict. If flatenned, names are used to determine alphabetical order.
        self.observation_space = spaces.Dict({
            "a.player" : spaces.Box(low=single_low_obs, high=single_high_obs, shape=(single_obs_size,), dtype=np.int32),
            "b.gate" : spaces.Box(low=single_low_obs, high=single_high_obs, shape=(single_obs_size,), dtype=np.int32),
            "c.obstacles" : spaces.Box(low=obstacles_low_obs, high=obstacles_high_obs, shape=(obstacles_obs_size,), dtype=np.int32),
            "d.turrets" : spaces.Box(low=turrets_low_obs, high=turrets_high_obs, shape=(turrets_obs_size,), dtype=np.int32),
            "e.projectiles" : spaces.Box(low=proj_low_obs, high=proj_high_obs, shape=(proj_obs_size,), dtype=np.int32),
            # "z.step" : spaces.Box(low=0, high=self.options.max_steps, shape=(1,), dtype=np.int32)
        })
        
        assert render_mode is None or render_mode in Game.metadata["render.modes"]
        self.render_fps = render_fps if render_fps else Game.metadata["default_render_fps"]
        self.render_mode = render_mode

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.all_sprites = pygame.sprite.Group()
        
        self.player = Player(self.options, [])
        self.all_sprites.add(self.player)

        self.gate = Gate(self.options, self.all_sprites)
        self.all_sprites.add(self.gate)
        
        self.turrets = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.turrets_and_obstacles = pygame.sprite.Group()
        if self.options.instantiate_turrets:
            for _ in range(self.options.n_turrets):
                t = Turret(self.options, self.all_sprites)
                self.turrets.add(t)
                self.all_sprites.add(t)
        if self.options.instantiate_obstacles:
            for _ in range(self.options.n_obstacles):
                o = Obstacle(self.options, self.all_sprites)
                self.obstacles.add(o)
                self.all_sprites.add(o)

        self.turrets_and_obstacles.add(self.turrets)
        self.turrets_and_obstacles.add(self.obstacles)
        self.all_projectiles = pygame.sprite.Group()

        self.prev_distance_to_gate = self._player_distance_to_gate()
        self.min_steps = int(self._player_distance_to_gate() / self.player.speed)

        self.clock = pygame.time.Clock()

        self.running = True
        self.step_count = 0
        self.score = 0
        self.reward = 0
        self.is_success = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _player_distance_to_gate(self):
        return np.linalg.norm(np.array(self.player.rect.center) - np.array(self.gate.rect.center))

    def _normalized_player_distance_to_gate(self):
        distance = self._player_distance_to_gate()
        return distance / (math.sqrt(self.options.width**2 + self.options.height**2))
    
    def _get_info(self):
        return {"score": self.score, 'is_success': self.is_success}

    def _get_obs(self):
        # for each object return top left and bottom right coordinates
        return {
            "a.player" : np.asarray([self.player.rect.left, self.player.rect.top, self.player.rect.right, self.player.rect.bottom], dtype=np.int32).flatten(),
            "b.gate" : np.asarray([self.gate.rect.left, self.gate.rect.top, self.gate.rect.right, self.gate.rect.bottom], dtype=np.int32).flatten(),
            "c.obstacles" : self._get_obstacles_obs(),
            "d.turrets" : self._get_turrets_obs(),
            "e.projectiles" : self._get_projectiles_obs(),
            # "z.step" : np.asarray([self.step_count+1], dtype=np.int32)
        }

    def _get_turrets_obs(self):
        # function that returns top left and bottom right coordinates of all turrets
        # If less than N_TURRETS, then fill with zeros
        turrets = []
        for turret in self.turrets:
            turrets.append(turret)
        for _ in range(self.options.n_turrets - len(turrets)):
            turrets.append(Turret(self.options,[], dummy=True))
        return np.asarray([[turret.rect.left, turret.rect.top, turret.rect.right, turret.rect.bottom] for turret in turrets], dtype=np.int32).flatten()

    def _get_obstacles_obs(self):
        # function that returns top left and bottom right coordinates of all obstacles
        # If less than N_OBSTACLES, then fill with zeros
        obstacles = []
        for obstacle in self.obstacles:
            obstacles.append(obstacle)
        for _ in range(self.options.n_obstacles - len(obstacles)):
            obstacles.append(Obstacle(self.options,[], dummy=True))
        return np.asarray([[obstacle.rect.left, obstacle.rect.top, obstacle.rect.right, obstacle.rect.bottom] for obstacle in obstacles], dtype=np.int32).flatten()

    # function that returns top left and bottom right coordinates of all projectiles
    # If less than MAX_PROJECTILES_PER_TURRET * N_TURRETS, then fill with zeros
    def _get_projectiles_obs(self):
        projectiles = []
        for projectile in self.all_projectiles:
            projectiles.append(projectile)
        for _ in range(self.options.max_projectiles_per_turret * self.options.n_turrets - len(projectiles)):
            projectiles.append(Projectile(0,0, self.options, dummy=True))
        return np.asarray([[projectile.rect.top, projectile.rect.left, projectile.rect.bottom, projectile.rect.right] for projectile in projectiles], dtype=np.int32).flatten()
        
    
    def get_actions_from_keys(self):
        pressed_keys = pygame.key.get_pressed()
        actions = self.get_noop_actions()
        if pressed_keys[K_UP]:
            actions[0]=0
        if pressed_keys[K_DOWN]:
            actions[0]=1
        if pressed_keys[K_LEFT]:
            actions[1]=0
        if pressed_keys[K_RIGHT]:
            actions[1]=1
        return actions

    def get_noop_actions(self):
        return np.asarray([2,2], dtype=np.int32)

    def _fire_turret(self):
        pygame.event.post(pygame.event.Event(Game.FIRE_TURRET))

    def process_events(self):
        for event in pygame.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop.
                if event.key == K_ESCAPE:
                    self.running = False

            # Did the user click the window close button? If so, stop the loop.
            elif event.type == QUIT:
                self.running = False
            elif event.type == Game.FIRE_TURRET:
                    for t in self.turrets:
                        if len(self.all_projectiles) < self.options.max_projectiles_per_turret * self.options.n_turrets:
                            projectile = t.fire(self.player.rect, self.player.motion_vector)
                            self.all_projectiles.add(projectile)

    def step(self, actions):

        self.process_events()

        # Fire turrent every time this function is called FIRE_TURRET_STEP_DELAY times
        # This is to make sure that the turrets fire at a constant rate regardless of FPS
        # Fire rate should be adjusted depending on the FPS.
        if self.options.fire_turret_step_delay!=0:
            if self.step_count % self.options.fire_turret_step_delay == 0:
                self._fire_turret()
        
        self.collision = self.player.update(actions, self.turrets_and_obstacles)
        for t in self.turrets: t.update(self.obstacles)

        self.all_sprites.add(self.all_projectiles)
        
        done = False
        self.reward = 0
        done = self._compute_reward()
        self.score += self.reward
        
        observation = self._get_obs()
        info = self._get_info()
        self.step_count += 1

        return (observation, self.reward, done, False, info)

    def _compute_reward(self):
        
        done = False
        # End condition 1: Player dies
        if pygame.sprite.spritecollideany(self.player, self.all_projectiles):
            self.player.kill()
            done = True
            self.is_success = False
            self.reward = self.options.rew.kill_penalty
        # End condition 2: Player reaches the gate
        elif pygame.sprite.spritecollideany(self.player, [self.gate]):
            done = True
            self.is_success = True
            self.reward = self.options.rew.win_reward
            # additional reward for finishing the level faster
            self.reward += (self.options.max_steps - self.step_count)*self.options.rew.remaining_step_reward_factor

        # Hitting obstacles or turrets
        elif self.collision:
            # End condition 3: Player hits an obstacle and touch_obstacle_kill is True
            if self.options.touch_obstacle_kill:
                done = True
                self.is_success = False
                self.reward = self.options.rew.kill_penalty
            else:
                self.reward = self.options.rew.collision_penalty
        
        # Reward for getting closer to the gate, penalize for getting further away or staying still
        else:
            delta_distance = self.prev_distance_to_gate - self._player_distance_to_gate()
            self.prev_distance_to_gate = self._player_distance_to_gate()
            if delta_distance>0:
                self.reward = delta_distance * self.options.rew.closer_reward_factor + self.options.rew.closer_reward_offset
            elif delta_distance==0:
                self.reward = self.options.rew.still_penalty
            else:
                self.reward = delta_distance * self.options.rew.further_penalty_factor + self.options.rew.further_penalty_offset
        return done


    def render(self):

        if self.render_mode == None:
            return None

        canvas = pygame.Surface((self.options.width, self.options.height))
        canvas.fill((0, 0, 0))
        for t in self.turrets: 
            t.blit(canvas)
        for o in self.obstacles: 
            o.blit(canvas)
        self.player.blit(canvas)
        self.gate.blit(canvas)
        
        if self.render_mode == "human":
            reward_img = self.font.render("Reward: {}".format(self.reward), True, (255, 255, 255))
            score_img = self.font.render("Score: {}".format(self.score), True, (255, 255, 255))
            fps_img = self.font.render("FPS: {}".format(self.clock.get_fps()), True, (255, 255, 255))
            canvas.blit(reward_img, (10, self.options.height-40))
            canvas.blit(score_img, (10, self.options.height-20))
            canvas.blit(fps_img, (10, self.options.height-60))
            self.screen.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        

    def close(self):
        pygame.quit()

    def run_loop(self):
        prev_time = time.time()
        while self.running:
            actions = self.get_actions_from_keys()
            obs, reward, done, _, info = self.step(actions)
            self.render()
            if done:
                print ("Done, success: %s, score: %s, length: %s" % (self.is_success, self.score, self.step_count))
                self.reset()
            # compute and print fps without using pygame clock
            fps = 1.0/(time.time()-prev_time)
            prev_time = time.time()
            print ("FPS: %s" % fps, end="\r", flush=True)
            
        self.close()