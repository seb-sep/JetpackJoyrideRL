class ConnectionHistory:
    def __init__(self, from_node, to_node, innovation_number, innovation_numbers):
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_number = innovation_number
        self.innovation_numbers = list(innovation_numbers)

    def matches(self, genome, from_node, to_node):
        if len(genome.genes) == len(self.innovation_numbers):
            if from_node.number == self.from_node and to_node.number == self.to_node:
                for gene in genome.genes:
                    if gene.innovation_no not in self.innovation_numbers:
                        return False
                return True
        return False
