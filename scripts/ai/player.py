import numpy as np
from scripts.ai.genome import Genome
import scripts.settings as settings
import pygame



class Player:
    def __init__(self):
        self.fitness = 0.0
        
        self.unadjusted_fitness = 0.0
        self.lifespan = 0
        self.best_score = 0
        self.score = 0
        self.gen = 0
        self.genome_inputs = 7
        self.genome_outputs = 2
        self.brain = Genome(self.genome_inputs, self.genome_outputs)  # Assuming Genome is defined elsewhere
        self.vision = np.zeros(self.genome_inputs)
        self.decision = np.zeros(self.genome_outputs)
        
        # State variables from the game
        self.is_moving_up = False
        self.dead = False
        self.died_by = None  # Possibilities 'eletricity' and 'rocket'
        self.paused = False
        self.player_pos_y = 645
        self.player_pos_x = 200
        self.player_vel_x = settings.DEFAULT_X_VELOCITY  # TODO make game progressive faster
        self.player_vel_y = 0
        
        self.player_fly_surface = pygame.image.load('assets/sprites/skins/PlayerFly_Blue.png').convert_alpha()
        self.player_fly_surface = pygame.transform.scale(self.player_fly_surface, [64, 68])  # 1.2x

        self.player_dead_surface = pygame.image.load('assets/sprites/skins/PlayerDead_Blue.png').convert_alpha()
        self.player_dead_surface = pygame.transform.scale(self.player_dead_surface, [82, 74])

        self.player_surface = self.player_fly_surface  # default player sprite
        self.player_rect = self.player_surface.get_rect(center=(256, 360))  # return new rectangle covering entire surface
        
        ### Events ###
        self.DIED = pygame.USEREVENT + 1  # event id
        self.died = pygame.event.Event(self.DIED)  # event object
        
        self.gravity = 1.2
        self.run_count = -5
        # self.size = 20
    

    def show(self, screen):
        # Draw the player on the screen
        player_image = self.player_fly_surface if not self.dead else self.player_dead_surface
        # screen.blit(player_image, (self.player_pos_x, self.player_pos_y))
        screen.blit(player_image, (self.player_pos_x, self.player_pos_y))
        # print("Player pos: ", self.player_pos_x, self.player_pos_y)
        

    def move(self, game_speed, main):
        # Movement logic goes here. Adjust posY and check for collisions.
        
        dt = main.dt
        
        # change player velocity (up || down) -change faster if going faster
        if not self.is_moving_up:
            self.player_vel_y += self.gravity * dt * self.player_vel_x * 1.8
        else:
            self.player_vel_y -= self.gravity * dt * self.player_vel_x * 1.8

        # keep player inside the bound
        if self.player_pos_y < settings.MAX_HEIGHT:  # if touch celling
            self.player_pos_y = settings.MAX_HEIGHT
            self.player_vel_y = 0
        elif self.player_pos_y > settings.MIN_HEIGHT - self.player_surface.get_size()[0]:  # if touch ground
            self.player_pos_y = settings.MIN_HEIGHT - self.player_surface.get_size()[0]
            self.player_vel_y = 0
        else:
            self.player_pos_y += self.player_vel_y * dt

        # update player position and draw
        self.player_rect.y = self.player_pos_y
        self.player_rect.x = self.player_pos_x
        main.screen.blit(self.player_surface, self.player_rect)

        # increase distance
        if not self.dead:
            self.score += dt * 0.05 * self.player_vel_x  # TODO change this to foreground x pos

    def update(self, dt, game_speed):
        self.move(dt, game_speed)

    def look(self, obstacles, game_speed):
        # Sensory input processing logic goes here.
        
        # Filter obstacles for only the ones in front of the player's current x position
            
        ###! Maybe pre-filter the obstacles !###
        # obs = filter(obstacles)
        
        ### Inputs to find ###
        
        ## Player Variables ##
            # Player Y position
            # X pos
            # X and y velocities
            
        ## Obstacle Variables ##
            # Find distance to the closest obstacle
            # Height of the closest obstacle
            # Width of the closest obstacle
            # Y position of closest obstacle
        
            # Find the gap between the current and next closest obstacles
            # X distance 
            # Y distance
        gap_between = settings.OBSTACLE_OFFSET 
        dist_to_obs_x = 999
        dist_to_obs_y = 999
        
        if len(obstacles) > 0:
            dist_to_obs_x = obstacles[0].x - self.player_pos_x
            dist_to_obs_y = obstacles[0].y - self.player_pos_y
            
        vision = [self.player_pos_x,
                  self.player_pos_y, 
                  self.player_vel_x, 
                  self.player_vel_y, 
                  dist_to_obs_x,
                  dist_to_obs_y,
                  gap_between]
        
        self.vision = vision
        

    def think(self):
        # Neural network decision-making logic goes here.
        max = 0
        # print("Vision", self.vision)
        # decision = self.brain.feed_forward2(self.vision)
        decision = self.brain.feed_forward(self.vision)
        # print(decision)
        
        self.is_moving_up = True if decision == 1 else False
        
    def calculate_fitness(self):
        self.fitness = self.score**2

    def clone(self):
        clone = Player()
        clone.brain = self.brain.clone()
        clone.brain.generate_network()
        clone.fitness = self.fitness
        clone.gen = self.gen
        clone.best_score = self.best_score
        return clone
        
        
    def crossover(self, parent2):
        child = Player()
        child.brain = self.brain.crossover(parent2.brain)
        child.brain.generate_network()
        return child
        


# Add other methods like cloneForReplay, , crossover, updateLocalObstacles as per your game logic.
