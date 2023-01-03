# Import and initialize the pygame library
import pygame
import gym
import math
from gym import spaces
import numpy as np
from sprites import Player, Turret, Obstacle, Projectile, Gate, Bounds

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

class GameOptions():
    
    def __init__(self, height = 800, 
                width=600, 
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
                reward_type = 2,
                instantiate_turrets = True,
                instantiate_obstacles = True
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
        self.reward_type = reward_type
        self.instantiate_turrets = instantiate_turrets
        self.instantiate_obstacles = instantiate_obstacles
        
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
    
    # Returns a dictionary of all options
    def get_params(self):
        return {
            "height" : self.height,
            "width" : self.width,
            "n_turrets" : self.n_turrets,
            "n_obstacles" : self.n_obstacles,
            "max_projectiles_per_turret" : self.max_projectiles_per_turret,
            "fire_turret_step_delay" : self.fire_turret_step_delay,
            "projectile_speed" : self.projectile_speed,
            "turret_bounds" : {
                "x" : self.turret_bounds.x,
                "y" : self.turret_bounds.y,
                "width" : self.turret_bounds.width,
                "height" : self.turret_bounds.height
            },
            "obstacle_bounds" : {
                "x" : self.obstacle_bounds.x,
                "y" : self.obstacle_bounds.y,
                "width" : self.obstacle_bounds.width,
                "height" : self.obstacle_bounds.height
            },
            "gate_bounds" : {
                "x" : self.gate_bounds.x,
                "y" : self.gate_bounds.y,
                "width" : self.gate_bounds.width,
                "height" : self.gate_bounds.height
            },
            "player_bounds" : {
                "x" : self.player_bounds.x,
                "y" : self.player_bounds.y,
                "width" : self.player_bounds.width,
                "height" : self.player_bounds.height
            },
            "max_steps" : self.max_steps,
            "reward_type" : self.reward_type,
            "instantiate_turrets" : self.instantiate_turrets,
            "instantiate_obstacles" : self.instantiate_obstacles
        }

    def __str__(self):
        return str(self.get_params())

    # creates Options instance from dictionary
    def from_params(params):
        height = params["height"]
        width = params["width"]
        n_turrets = params["n_turrets"]
        n_obstacles = params["n_obstacles"]
        max_projectiles_per_turret = params["max_projectiles_per_turret"]
        fire_turret_step_delay = params["fire_turret_step_delay"]
        max_projectiles_per_turret = params["max_projectiles_per_turret"]
        fire_turret_step_delay = params["fire_turret_step_delay"]
        projectile_speed = params["projectile_speed"]
        turret_bounds = Bounds(params["turret_bounds"]["x"], params["turret_bounds"]["y"], params["turret_bounds"]["width"], params["turret_bounds"]["height"])
        obstacle_bounds = Bounds(params["obstacle_bounds"]["x"], params["obstacle_bounds"]["y"], params["obstacle_bounds"]["width"], params["obstacle_bounds"]["height"])
        gate_bounds = Bounds(params["gate_bounds"]["x"], params["gate_bounds"]["y"], params["gate_bounds"]["width"], params["gate_bounds"]["height"])
        player_bounds = Bounds(params["player_bounds"]["x"], params["player_bounds"]["y"], params["player_bounds"]["width"], params["player_bounds"]["height"])
        max_steps = params["max_steps"]
        reward_type = params["reward_type"]
        instantiate_turrets = params["instantiate_turrets"]
        instantiate_obstacles = params["instantiate_obstacles"]
        return GameOptions(height, width, n_turrets, n_obstacles, max_projectiles_per_turret, fire_turret_step_delay, projectile_speed, turret_bounds, obstacle_bounds, gate_bounds, player_bounds, max_steps, reward_type, instantiate_turrets, instantiate_obstacles)

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

        max_coordinate = max(self.options.width, self.options.height)
        N_COORD = 4
        self.observation_space = spaces.Dict({
            "a.player" : spaces.Box(low=0, high=max_coordinate, shape=(N_COORD,), dtype=np.int32),
            "d.turrets" : spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_turrets*N_COORD,), dtype=np.int32),
            "c.obstacles" : spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_obstacles*N_COORD,), dtype=np.int32),
            "b.gate" : spaces.Box(low=0, high=max_coordinate, shape=(N_COORD,), dtype=np.int32),
            "e.projectiles" : spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_turrets*self.options.max_projectiles_per_turret*N_COORD,), dtype=np.int32),
            # "z.step" : spaces.Box(low=0, high=self.options.max_steps, shape=(1,), dtype=np.int32) # calling it zstep so that flatten wrapper puts it last
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

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _player_distance_to_gate(self):
        return np.linalg.norm(np.array(self.player.rect.center) - np.array(self.gate.rect.center))

    def _normalized_player_distance_to_gate(self):
        distance = self._player_distance_to_gate()
        return distance / (math.sqrt(self.options.width**2 + self.options.height**2))
    
    def _get_info(self):
        return {"score": self.score}

    def _get_obs(self):
        # for each object return top left and bottom right coordinates
        return {
            "a.player" : np.asarray([self.player.rect.top, self.player.rect.left, self.player.rect.bottom, self.player.rect.right], dtype=np.int32).flatten(),
            "d.turrets" : self._get_turrets_obs(),
            "c.obstacles" : self._get_obstacles_obs(),
            "b.gate" : np.asarray([self.gate.rect.top, self.gate.rect.left, self.gate.rect.bottom, self.gate.rect.right], dtype=np.int32).flatten(),
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
        return np.asarray([[turret.rect.top, turret.rect.left, turret.rect.bottom, turret.rect.right] for turret in turrets], dtype=np.int32).flatten()

    def _get_obstacles_obs(self):
        # function that returns top left and bottom right coordinates of all obstacles
        # If less than N_OBSTACLES, then fill with zeros
        obstacles = []
        for obstacle in self.obstacles:
            obstacles.append(obstacle)
        for _ in range(self.options.n_obstacles - len(obstacles)):
            obstacles.append(Obstacle(self.options,[], dummy=True))
        return np.asarray([[obstacle.rect.top, obstacle.rect.left, obstacle.rect.bottom, obstacle.rect.right] for obstacle in obstacles], dtype=np.int32).flatten()

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
            self.reward = -100
        # End condition 2: Player reaches the gate
        elif pygame.sprite.spritecollideany(self.player, [self.gate]):
            done = True
            self.reward = 500
            # reamining steps are added to the reward
            self.reward += self.options.max_steps - self.step_count

        # Penalize hitting obstacles or turrets
        elif self.collision:
            self.reward = -1
        
        # Reward for getting closer to the gate, penalize for getting further away or staying still
        else:
            delta_distance = self.prev_distance_to_gate - self._player_distance_to_gate()
            self.prev_distance_to_gate = self._player_distance_to_gate()
            if delta_distance>0:
                self.reward = delta_distance
            else:
                self.reward = -1
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
        while self.running:
            actions = self.get_actions_from_keys()
            obs, reward, done, _, info = self.step(actions)
            self.render()
            if done:
                print ("Done, reward: %s, score: %s" % (reward, self.score))
                self.reset()
        self.close()