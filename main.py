# Import and initialize the pygame library
import pygame
import random
import math
import logging
from pygame.math import Vector2

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

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((25, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
        self.rect.left = SCREEN_WIDTH/2
        self.rect.bottom = SCREEN_HEIGHT*0.9
        self.motion_vector = Vector2(0,0)
        self.speed = 7

    def update(self, actions, obstacles):
        prev_pos= self.rect.center
        if actions[Game.UP]:
            self.rect.move_ip(0, -self.speed)
        if actions[Game.DOWN]:
            self.rect.move_ip(0, self.speed)
        if actions[Game.LEFT]:
            self.rect.move_ip(-self.speed, 0)
        if actions[Game.RIGHT]:
            self.rect.move_ip(self.speed, 0)

        if pygame.sprite.spritecollideany(self, obstacles):
            self.rect.center = prev_pos
    
        self.motion_vector = Vector2(self.rect.center) - Vector2(prev_pos)
                    
        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

    def blit(self, screen):
        screen.blit(self.surf, self.rect)

def normalized_vector (point_a, point_b):
    return (Vector2(point_b)-Vector2(point_a)).normalize()

class Turret(pygame.sprite.Sprite):
    def __init__(self):
        super(Turret, self).__init__()
        self.surf = pygame.Surface((15, 15))
        #self.surf.fill((0, 255, 255))
        self.rect = self.surf.get_rect()
        pygame.draw.circle(self.surf, (0, 255, 255),self.rect.center, 7.5)
        

        self.rect.left = SCREEN_WIDTH*random.uniform(0,1)
        self.rect.bottom = SCREEN_HEIGHT*random.uniform(0,0.7)
        self.projectiles = pygame.sprite.Group()

    def fire(self, target, target_motion_vector):
        direction = normalized_vector(self.rect.center, Vector2(target.center)+target_motion_vector)
        print("target: %s, origin: %s, direction: %s"%(target, self.rect.center,direction))
        p = Projectile(self.rect.top, self.rect.left, direction)
        self.projectiles.add(p)

    def update(self, obstacles):
        for p in self.projectiles:
            p.update()
            p.check_obstacle_collision(obstacles)

    def blit(self, screen):
        for p in self.projectiles:
            p.blit(screen)
        screen.blit(self.surf, self.rect)



class Projectile(pygame.sprite.Sprite):
    def __init__(self, init_top, init_left, direction=(0,3)):
        super(Projectile, self).__init__()
        self.surf = pygame.Surface((5, 5))
        self.surf.fill((0, 0, 255))
        self.pos = (init_left, init_top)
        self.rect = self.surf.get_rect()
        self.rect.center = self.pos
        self.direction = direction
        self.speed = 10

    def update(self):
        self.pos = (self.pos[0] + self.speed*self.direction[0], self.pos[1] + self.speed*self.direction[1])
        self.rect.center = self.pos
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.kill()

    def check_obstacle_collision(self, obstacles):
        if pygame.sprite.spritecollideany(self, obstacles):
            self.kill()
    
    def blit(self, screen):
        screen.blit(self.surf, self.rect)


class Obstacle(pygame.sprite.Sprite):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.surf = pygame.Surface((45, 45))
        self.rect = self.surf.get_rect()
        pygame.draw.polygon(self.surf, (0, 255, 0),[
            (self.rect.left,self.rect.bottom),
            (self.rect.right, self.rect.bottom),
            ((self.rect.right+self.rect.left)/2,self.rect.top)
            ])
            
        self.rect.left = SCREEN_WIDTH*random.uniform(0,1)
        self.rect.bottom = SCREEN_HEIGHT*random.uniform(0,0.8)

    def blit(self, screen):
        screen.blit(self.surf, self.rect)

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800

class Game():

    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

    # Define constants for the screen width and height
    N_TURRETS = 5
    N_OBSTACLES = 3

    FIRE_TURRET = pygame.USEREVENT + 1

    def __init__(self) -> None:

        pygame.init()

        # Set up the drawing window
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.player = Player()
        self.turrets = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.turrets_and_obstacles = pygame.sprite.Group()
        for _ in range(Game.N_TURRETS):
            self.turrets.add(Turret())
        for _ in range(Game.N_OBSTACLES):
            self.obstacles.add(Obstacle())

        self.turrets_and_obstacles.add(self.turrets)
        self.turrets_and_obstacles.add(self.obstacles)

        pygame.time.set_timer(Game.FIRE_TURRET, 2000)

        self.clock = pygame.time.Clock()

        self.running = True

    def get_new_actions_array(self):
        return {Game.UP:False, Game.DOWN:False, Game.LEFT:False, Game.RIGHT:False}
    
    def get_actions_from_keys(self):
        pressed_keys = pygame.key.get_pressed()
        actions = self.get_new_actions_array()
        if pressed_keys[K_UP]:
            actions[Game.UP]=True
        if pressed_keys[K_DOWN]:
            actions[Game.DOWN]=True
        if pressed_keys[K_LEFT]:
            actions[Game.LEFT]=True
        if pressed_keys[K_RIGHT]:
            actions[Game.RIGHT]=True
        return actions

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
        self.player.update(actions, self.turrets_and_obstacles)
        for t in self.turrets: t.update(self.obstacles)
        all_projectiles=pygame.sprite.Group()
        for t in self.turrets: 
            all_projectiles.add(t.projectiles)
        if pygame.sprite.spritecollideany(self.player, all_projectiles):
            # If so, then remove the player and stop the loop
            #player.kill()
            #self.running = False
            pass
        self._render()

    def _render(self):
        self.screen.fill((0, 0, 0))
        for t in self.turrets: 
            t.blit(self.screen)
        for o in self.obstacles: 
            o.blit(self.screen)
        self.player.blit(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close():
        pygame.quit()

    def game_mode_loop(self):
        while self.running:
            self.process_events()
            actions = self.get_actions_from_keys()
            self.step(actions)
        self.close()



game = Game()
game.game_mode_loop()
    