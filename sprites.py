import pygame
from pygame.math import Vector2
import random


class Player(pygame.sprite.Sprite):
    def __init__(self, options):
        super(Player, self).__init__()
        self.options = options
        self.surf = pygame.Surface((25, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
        self.rect.left = self.options.width/2
        self.rect.bottom = self.options.height*0.9
        self.motion_vector = Vector2(0,0)
        self.speed = 7

    def update(self, actions, obstacles):
        prev_pos= self.rect.center
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

    def blit(self, screen):
        screen.blit(self.surf, self.rect)

def normalized_vector (point_a, point_b):
    return (Vector2(point_b)-Vector2(point_a)).normalize()

class Turret(pygame.sprite.Sprite):
    def __init__(self, options):
        super(Turret, self).__init__()
        self.options = options
        self.surf = pygame.Surface((15, 15))
        self.rect = self.surf.get_rect()
        pygame.draw.circle(self.surf, (0, 255, 255),self.rect.center, 7.5)
        

        self.rect.center = (self.options.width*random.uniform(0.05,0.95), self.options.height*random.uniform(0.05,0.7))
        self.projectiles = pygame.sprite.Group()

    def fire(self, target, target_motion_vector):
        direction = normalized_vector(self.rect.center, Vector2(target.center)+target_motion_vector)
        p = Projectile(self.rect.top, self.rect.left, self.options, direction)
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
    def __init__(self, init_top, init_left, options, direction=(0,3)):
        super(Projectile, self).__init__()
        self.options = options
        self.surf = pygame.Surface((5, 5))
        self.surf.fill((230, 250, 0))
        self.pos = (init_left, init_top)
        self.rect = self.surf.get_rect()
        self.rect.center = self.pos
        self.direction = direction
        self.speed = 10

    def update(self):
        self.pos = (self.pos[0] + self.speed*self.direction[0], self.pos[1] + self.speed*self.direction[1])
        self.rect.center = self.pos
        if self.rect.bottom >= self.options.height or self.rect.top <= 0 or self.rect.left <= 0 or self.rect.right >= self.options.width:
            self.kill()

    def check_obstacle_collision(self, obstacles):
        if pygame.sprite.spritecollideany(self, obstacles):
            self.kill()
    
    def blit(self, screen):
        screen.blit(self.surf, self.rect)


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, options):
        super(Obstacle, self).__init__()
        self.options = options
        self.surf = pygame.Surface((45, 45))
        self.rect = self.surf.get_rect()
        pygame.draw.polygon(self.surf, (0, 255, 0),[
            (self.rect.left,self.rect.bottom),
            (self.rect.right, self.rect.bottom),
            ((self.rect.right+self.rect.left)/2,self.rect.top)
            ])
            
        self.rect.center = (self.options.width*random.uniform(0,1), self.options.height*random.uniform(0,0.8))

    def blit(self, screen):
        screen.blit(self.surf, self.rect)

class Gate(pygame.sprite.Sprite):
    def __init__(self, options):
        super(Gate, self).__init__()
        self.options = options
        self.surf = pygame.Surface((self.options.width/3, 20))
        self.rect = self.surf.get_rect()
        self.surf.fill((100, 100, 100))
        self.rect.center = (self.options.width/2, 0)
        self.rect.top = 0
    
    def blit(self, screen):
        screen.blit(self.surf, self.rect)
