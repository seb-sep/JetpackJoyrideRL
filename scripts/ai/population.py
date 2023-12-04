import numpy as np
import random
from scripts.ai.player import Player

class Population:
    def __init__(self, size):
        self.pop = [Player() for _ in range(size)]

        self.best_player = None
        self.best_score = 0
        self.gen = 0
        self.innovation_history = []
        self.gen_players = []
        self.species = []
        self.mass_extinction_event = False
        self.new_stage = False
        self.population_life = 0

        for player in self.pop:
            player.brain.generate_network()
            player.brain.mutate(self.innovation_history)
            
    def update_alive(self):
        self.population_life += 1
        for player in self.pop:
            if not player.dead:
                player.look()
                player.think()
                player.update()
                if not self.show_nothing:
                    player.show()

    def done(self):
        return all(player.dead for player in self.pop)

    def set_best_player(self):
        # Implementation depends on your specific logic for determining the best player
        pass

    def natural_selection(self):
        self.speciate()
        self.calculate_fitness()
        self.sort_species()
        # Additional methods and logic
        # ...
    
    def speciate(self):
        # Implementation depends on your specific logic for speciation
        pass

    def calculate_fitness(self):
        for player in self.pop:
            player.calculate_fitness()

    def sort_species(self):
        # Sort species by fitness or other criteria
        pass

    def mass_extinction(self):
        if len(self.species) > 5:
            self.species = self.species[:5]

    def get_avg_fitness_sum(self):
        return sum(species.average_fitness for species in self.species)

    def cull_species(self):
        for species in self.species:
            species.cull()
            species.fitness_sharing()
            species.set_average()

    def kill_stale_species(self):
        self.species = [s for s in self.species if s.staleness < 15]

    def kill_bad_species(self):
        average_sum = self.get_avg_fitness_sum()
        self.species = [s for s in self.species if len(s.players) / average_sum * len(self.pop) >= 1]
        
    # Checks if all the players in the population have died
    def all_players_dead(self):
        for player in self.pop:
            if player.dead == False: return False
        
        return True