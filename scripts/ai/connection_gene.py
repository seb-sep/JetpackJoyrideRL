import numpy as np

class ConnectionGene:
    def __init__(self, from_node, to_node, weight, innovation_no):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = True
        self.innovation_no = innovation_no

    def mutate_weight(self):
        rand2 = np.random.rand()
        if rand2 < 0.1:  # 10% chance to completely change the weight
            self.weight = np.random.uniform(-1, 1)
        else:  # Otherwise slightly change it
            self.weight += np.random.normal() / 50
            # Keep weight within bounds
            self.weight = max(min(self.weight, 1), -1)

    def clone(self, from_node=None, to_node=None):
        if from_node is None:
            from_node = self.from_node
        if to_node is None:
            to_node = self.to_node
        clone = ConnectionGene(from_node, to_node, self.weight, self.innovation_no)
        clone.enabled = self.enabled
        return clone
