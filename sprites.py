import pygame
from pygame.math import Vector2
import random


class Bounds(pygame.Rect):
    def __init__(self, left, top, width, height):
        super(Bounds, self).__init__(left, top, width, height)
    def __hash__(self):
        return hash((self.left, self.top, self.width, self.height))

class BaseSprite(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super(BaseSprite, self).__init__()

    def blit(self, screen):
        screen.blit(self.surf, self.rect)

class BaseRandomSprite(BaseSprite):
    
    def __init__(self, size, bounds, other_sprites, dummy=False) -> None:
        super(BaseRandomSprite, self).__init__()
        # initialize randomly the location of the sprite within the bounds
        # and checking there is no collision with other sprites
        if dummy:
            self.surf = pygame.Surface((0,0))
            self.rect = self.surf.get_rect()
            self.rect.top = 0
            self.rect.left = 0
            return
        
        self.surf = pygame.Surface(size)
        self.rect = self.surf.get_rect()

        self.rect.top = random.randint(bounds.top, bounds.bottom-size[1])
        self.rect.left = random.randint(bounds.left, bounds.right-size[0])
        i = 0
        while pygame.sprite.spritecollideany(self, other_sprites):
            if i >= 100:
                raise Exception("Unable to place sprite")
            i = i + 1
            self.rect.top = random.randint(bounds.top, bounds.bottom-size[1])
            self.rect.left = random.randint(bounds.left, bounds.right-size[0])
        

PLAYER_SIZE = (25,25)
PLAYER_COLOR = (255, 255, 255)

class Player(BaseRandomSprite):
    def __init__(self, options, other_sprites, **kwargs):
        super(Player, self).__init__(PLAYER_SIZE, options.player_bounds, other_sprites, **kwargs)
        self.options = options
        self.surf.fill(PLAYER_COLOR)
        self.motion_vector = Vector2(0,0)
        self.speed = 7

    def update(self, actions, obstacles):
        prev_pos= self.rect.center
        collision = False
        if actions[0] == 0:
            self.rect.move_ip(0, -self.speed)
        if actions[0] == 1:
            self.rect.move_ip(0, self.speed)
        if actions[1] == 0:
            self.rect.move_ip(-self.speed, 0)
        if actions[1] == 1:
            self.rect.move_ip(self.speed, 0)

        if pygame.sprite.spritecollideany(self, obstacles):
            self.rect.center = prev_pos
            collision = True
    
        self.motion_vector = Vector2(self.rect.center) - Vector2(prev_pos)
                    
        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.options.width:
            self.rect.right = self.options.width
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= self.options.height:
            self.rect.bottom = self.options.height

        return collision

def normalized_vector (point_a, point_b):
    return (Vector2(point_b)-Vector2(point_a)).normalize()

TURRET_SIZE = (15,15)
TURRET_COLOR = (0, 255, 255)
class Turret(BaseRandomSprite):
    def __init__(self, options, all_turrets, **kwargs):
        super(Turret, self).__init__(TURRET_SIZE, options.turret_bounds, all_turrets, **kwargs)
        self.options = options
        pygame.draw.circle(self.surf, TURRET_COLOR,self.surf.get_rect().center, TURRET_SIZE[0]/2)
        self.projectiles = pygame.sprite.Group()

    def fire(self, target, target_motion_vector):
        direction = normalized_vector(self.rect.center, Vector2(target.center)+target_motion_vector)
        p = Projectile(self.rect.top, self.rect.left, self.options, direction)
        self.projectiles.add(p)
        return p

    def update(self, obstacles):
        for p in self.projectiles:
            p.update()
            p.check_obstacle_collision(obstacles)

    def blit(self, screen):
        for p in self.projectiles:
            p.blit(screen)
        super().blit(screen)



PROJECTILE_SIZE = (5,5)
PROJECTILE_COLOR = (230, 250, 0)
class Projectile(BaseSprite):
    def __init__(self, 
                init_top, 
                init_left, 
                options, 
                direction=(0,0),
                dummy=False,):
        super(Projectile, self).__init__()
        self.options = options
        if dummy:
            self.surf = pygame.Surface((0,0))
        else:
            self.surf = pygame.Surface(PROJECTILE_SIZE)
        self.surf.fill(PROJECTILE_COLOR)
        self.pos = (init_left, init_top)
        self.rect = self.surf.get_rect()
        self.rect.center = self.pos
        self.direction = direction

    def update(self):
        speed = self.options.projectile_speed
        self.pos = (self.pos[0] + speed*self.direction[0], self.pos[1] + speed*self.direction[1])
        self.rect.center = self.pos
        if self.rect.bottom >= self.options.height or self.rect.top <= 0 or self.rect.left <= 0 or self.rect.right >= self.options.width:
            self.kill()

    def check_obstacle_collision(self, obstacles):
        if pygame.sprite.spritecollideany(self, obstacles):
            self.kill()
    

OBSTACLE_SIZE = (45,45)
OBSTACLE_COLOR = (0, 255, 0)
class Obstacle(BaseRandomSprite):
    def __init__(self, 
                options, all_obstacles, **kwargs):
        super(Obstacle, self).__init__(OBSTACLE_SIZE, options.obstacle_bounds, all_obstacles, **kwargs)
        self.options = options
        surf_rect = self.surf.get_rect()
        pygame.draw.polygon(self.surf, OBSTACLE_COLOR,[
            (surf_rect.left,surf_rect.bottom),
            (surf_rect.right, surf_rect.bottom),
            ((surf_rect.right+surf_rect.left)/2,surf_rect.top)
            ])
            

GATE_SIZE = (60, 20)
GATE_COLOR = (100, 100, 100)
class Gate(BaseRandomSprite):
    def __init__(self, options, other_sprites):
        super(Gate, self).__init__(GATE_SIZE, options.gate_bounds, other_sprites)
        self.options = options
        self.surf.fill(GATE_COLOR)


