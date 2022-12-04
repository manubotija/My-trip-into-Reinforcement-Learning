# Import and initialize the pygame library
import pygame
import gym
import math
from gym import spaces
import numpy as np
from sprites import Player, Turret, Obstacle, Projectile, Gate

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
    
    def __init__(self, height = 800, width=600, n_turrets=3, 
                n_obstacles=3, max_projectiles_per_turret=5, 
                fire_turret_step_delay=45) -> None:
        self.height = height
        self.width = width
        self.n_turrets = n_turrets
        self.n_obstacles = n_obstacles
        self.max_projectiles_per_turret = max_projectiles_per_turret
        self.fire_turret_step_delay = fire_turret_step_delay
        self.max_projectiles_per_turret = max_projectiles_per_turret
        self.fire_turret_step_delay = fire_turret_step_delay
        

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

        self.action_space = spaces.MultiDiscrete([3, 3])

        max_coordinate = max(self.options.width, self.options.height)
        self.observation_space = spaces.Dict({
            "player" : spaces.Dict({
                "center" : spaces.Box(low=0, high=max_coordinate, shape=(2,), dtype=np.int32),
                "size" : spaces.Box(low=0, high=max_coordinate, shape=(2,), dtype=np.int32)
            }),
            "turrets" : spaces.Dict({
                "centers" : spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_turrets*2,), dtype=np.int32),
                "sizes" :   spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_turrets*2,), dtype=np.int32),
            }),
            "obstacles" : spaces.Dict({
                "centers" : spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_obstacles*2,), dtype=np.int32),
                "sizes" :   spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_obstacles*2,), dtype=np.int32),
            }),
            "gate" : spaces.Dict({
                "center" : spaces.Box(low=0, high=max_coordinate, shape=(2,), dtype=np.int32),
                "size" : spaces.Box(low=0, high=max_coordinate, shape=(2,), dtype=np.int32)
            }),
            "projectiles" : spaces.Dict({
                "centers" : spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_turrets*self.options.max_projectiles_per_turret*2,), dtype=np.int32),
                #"sizes" :   spaces.Box(low=0, high=max_coordinate, shape=(self.options.n_turrets*self.options.max_projectiles_per_turret, 2), dtype=np.int32),
            })
        })
        
        assert render_mode is None or render_mode in Game.metadata["render.modes"]
        self.render_fps = render_fps if render_fps else Game.metadata["default_render_fps"]
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player = Player(self.options)
        self.gate = Gate(self.options)
        self.turrets = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.turrets_and_obstacles = pygame.sprite.Group()
        for _ in range(self.options.n_turrets):
            self.turrets.add(Turret(self.options))
        for _ in range(self.options.n_obstacles):
            self.obstacles.add(Obstacle(self.options))

        self.turrets_and_obstacles.add(self.turrets)
        self.turrets_and_obstacles.add(self.obstacles)
        self.all_projectiles = pygame.sprite.Group()

        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add([self.player, self.gate, self.turrets_and_obstacles])

        self.clock = pygame.time.Clock()

        self.running = True
        self.step_count = 0
        self.score = 0
        self.reward = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    # computes the distance between the center of the gate and the center of the player
    def _normalized_player_distance_to_gate(self):
        distance = np.linalg.norm(np.array(self.player.rect.center) - np.array(self.gate.rect.center))
        return distance / (math.sqrt(self.options.width**2 + self.options.height**2))
    
    def _get_info(self):
        return {"score": self.score}

    def _get_obs(self):
        return {
            "player" : {
                "center" : np.asarray(self.player.rect.center, dtype=np.int32),
                "size" : np.asarray(self.player.rect.size, dtype=np.int32)
            },
            "turrets" : {
                "centers" : np.asarray([turret.rect.center for turret in self.turrets], dtype=np.int32).flatten(),
                "sizes" : np.asarray([turret.rect.size for turret in self.turrets], dtype=np.int32).flatten()
            },
            "obstacles" : {
                "centers" : np.asarray([obstacle.rect.center for obstacle in self.obstacles], dtype=np.int32).flatten(),
                "sizes" : np.asarray([obstacle.rect.size for obstacle in self.obstacles], dtype=np.int32).flatten()
            },
            "gate" : {
                "center" : np.asarray(self.gate.rect.center, dtype=np.int32),
                "size" : np.asarray(self.gate.rect.size, dtype=np.int32)
            },
            "projectiles" : self._get_projectiles()
        }

    # function that returns Dict containing centers and sizes for all projectiles. 
    # If less than MAX_PROJECTILES_PER_TURRET * N_TURRETS, then fill with zeros
    def _get_projectiles(self):
        projectiles = []
        for projectile in self.all_projectiles:
            projectiles.append(projectile)
        for _ in range(self.options.max_projectiles_per_turret * self.options.n_turrets - len(projectiles)):
            projectiles.append(Projectile(0,0, self.options))
        return {            
                "centers" : np.asarray([projectile.rect.center for projectile in projectiles], dtype=np.int32).flatten(),
                #"sizes" : np.asarray([projectile.rect.size for projectile in projectiles], dtype=np.int32)
            }
        
    
    def get_actions_from_keys(self):
        pressed_keys = pygame.key.get_pressed()
        actions = np.zeros(2, dtype=np.int32)
        if pressed_keys[K_UP]:
            actions[0]=0
        if pressed_keys[K_DOWN]:
            actions[0]=1
        if pressed_keys[K_LEFT]:
            actions[1]=0
        if pressed_keys[K_RIGHT]:
            actions[1]=1
        return actions

    def _fire_turret(self):
        pygame.event.post(pygame.event.Event(Game.FIRE_TURRET))
        pass

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
                for t in self.turrets: t.fire(self.player.rect, self.player.motion_vector)

    def step(self, actions):

        self.process_events()

        # Fire turrent every time this function is called FIRE_TURRET_STEP_DELAY times
        # This is to make sure that the turrets fire at a constant rate regardless of FPS
        # Fire rate should be adjusted depending on the FPS.
        if self.options.fire_turret_step_delay!=0:
            if self.step_count % self.options.fire_turret_step_delay == 0:
                self._fire_turret()
        self.step_count += 1

        self.reward = 0
        done = False
        self.player.update(actions, self.turrets_and_obstacles)
        for t in self.turrets: t.update(self.obstacles)
        for t in self.turrets: 
            self.all_projectiles.add(t.projectiles)
        self.all_sprites.add(self.all_projectiles)
        if pygame.sprite.spritecollideany(self.player, self.all_projectiles):
            # If so, then remove the player and stop the loop
            self.player.kill()
            done = True
            self.reward += -200
            self.score += self.reward
        elif pygame.sprite.spritecollideany(self.player, [self.gate]):
            done = True
            self.reward += 1000
            self.score += self.reward
        else:
            #exponential reward for getting closer to the gate
            #self.reward += 100 * math.exp(-self._normalized_player_distance_to_gate()*10)
            #self.reward = (1 - self._normalized_player_distance_to_gate())*10
            self.reward = 1/(self._normalized_player_distance_to_gate()**2)
            #quantize reward
            self.reward = int(self.reward)*10
        
        
        observation = self._get_obs()
        info = self._get_info()

        return (observation, self.reward, done, False, info)


    def render(self):

        if self.render_mode == None:
            return

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