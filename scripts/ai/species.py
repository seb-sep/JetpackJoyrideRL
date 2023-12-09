import random
from scripts.ai.genome import Genome

class Species:
    def __init__(self, player=None):
        self.players = []
        self.best_fitness = 0
        self.champ = None
        self.average_fitness = 0
        self.staleness = 0
        self.rep = None
        self.excess_coeff = 1.0
        self.weight_diff_coeff = 0.5
        self.compatibility_threshold = 3.0

        if player is not None:
            self.players.append(player)
            self.best_fitness = player.fitness
            self.rep = player.brain.clone()
            # self.champ = player.clone_for_replay()

    def same_species(self, genome: Genome):
        excess_and_disjoint = self.get_excess_disjoint(genome, self.rep)
        average_weight_diff = self.average_weight_diff(genome, self.rep)

        large_genome_normaliser = len(genome.genes) - 20
        if large_genome_normaliser < 1:
            large_genome_normaliser = 1

        compatibility = (self.excess_coeff * excess_and_disjoint / large_genome_normaliser) + (self.weight_diff_coeff * average_weight_diff)
        return compatibility < self.compatibility_threshold

    def get_excess_disjoint(self, brain1: Genome, brain2: Genome):
        matching = 0.0
        for gene1 in brain1.genes:
            for gene2 in brain2.genes:
                if gene1.innovation_no == gene2.innovation_no:
                    matching += 1
                    break
        return len(brain1.genes) + len(brain2.genes) - 2 * matching

    def average_weight_diff(self, brain1, brain2):
        if len(brain1.genes) == 0 or len(brain2.genes) == 0:
            return 0

        matching = 0
        total_diff = 0
        for gene1 in brain1.genes:
            for gene2 in brain2.genes:
                if gene1.innovation_no == gene2.innovation_no:
                    matching += 1
                    total_diff += abs(gene1.weight - gene2.weight)
                    break
        if matching == 0:
            return 100
        return total_diff / matching

    def sort_species(self):
        self.players.sort(key=lambda x: x.fitness, reverse=True)
        if self.players:
            if self.players[0].fitness > self.best_fitness:
                self.staleness = 0
                self.best_fitness = self.players[0].fitness
                self.rep = self.players[0].brain.clone()
                self.champ = self.players[0].clone()
                # self.champ = self.players[0]
            else:
                self.staleness += 1

    def set_average(self):
        if self.players:
            self.average_fitness = sum(player.fitness for player in self.players) / len(self.players)

    def get_child(self, innovation_history):
        if random.random() < 0.25:
            child = self.select_player().clone()
        else:
            print("getting child")
            parent1 = self.select_player()
            parent2 = self.select_player()
            print("got parents")
            child = parent1.crossover(parent2) if parent1.fitness >= parent2.fitness else parent2.crossover(parent1)
            print("crossover")
            
        child.brain.mutate(innovation_history)
        return child

    def select_player(self):
        fitness_sum = sum(player.fitness for player in self.players)
        rand = random.uniform(0, fitness_sum)
        running_sum = 0
        for player in self.players:
            running_sum += player.fitness
            if running_sum > rand:
                return player
            
        print("Something went wrong, no player was selected")
        return self.players[0]

    def cull(self):
        if len(self.players) > 2:
            self.players = self.players[:len(self.players) // 2]

    def fitness_sharing(self):
        for player in self.players:
            player.fitness /= len(self.players)
