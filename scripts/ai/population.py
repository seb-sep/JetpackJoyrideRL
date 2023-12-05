import numpy as np
import random
from scripts.ai.player import Player
from scripts.ai.species import Species

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

        sum = 0
        for player in self.pop:
            player.brain.generate_network()
            player.brain.mutate(self.innovation_history)
            sum += player.fitness
            
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
        for player in self.pop:
            if player.score > self.best_score:
                self.best_score = player.score
                self.best_player = player


    def natural_selection(self):
        self.speciate()
        print('species', len(self.species))
        self.calculate_fitness()
        print('species', len(self.species))
        self.sort_species()
        print('species', len(self.species))
        # Additional methods and logic
        # self.kill_stale_species()
        # print('species', len(self.species))
        self.kill_bad_species()
        print('species', len(self.species))
        self.cull_species()
        print('species', len(self.species))
        self.mass_extinction()
        print('species', len(self.species))
        
        self.reproduce()
    
    def speciate(self):
        for player in self.pop:
            species_found = False
            for species in self.species:
                if species.same_species(player.brain):
                    species.players.append(player)
                    species_found = True
                    break
            if not species_found:
                self.species.append(Species(player))

    def calculate_fitness(self):
        for player in self.pop:
            player.calculate_fitness()

    def sort_species(self):
        # Sort species by fitness or other criteria
        self.species.sort(key=lambda s: s.average_fitness, reverse=True)

        

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
        [print(species.staleness) for species in self.species]
        self.species = [s for s in self.species if s.staleness < 15]

    def kill_bad_species(self):
        average_sum = self.get_avg_fitness_sum()
        # print(len(self.pop), average_sum)
        # only save the species which have a fitness greater than the average
        self.species = [s for s in self.species if s.average_fitness >= average_sum]
        # self.species = [s for s in self.species if len(s.players) / average_sum * len(self.pop) >= 1]
        
    # Checks if all the players in the population have died
    def all_players_dead(self):
        # using all() to shorten the code
        return all(player.dead for player in self.pop)
    

    def reproduce(self):
        new_population = []
        saved_players = sum([len(species.players) for species in self.species if len(species.players) > 1])
        for species in self.species:
            species.sort_species()
            # new_population.append(species.champ.clone())
            for _ in range(len(species.players) - 1):
                new_population.append(species.give_me_baby(self.innovation_history))
        while len(new_population) < len(self.pop) - saved_players:
            species = np.random.choice(self.species)
            new_population.append(species.give_me_baby(self.innovation_history))
        self.pop = new_population