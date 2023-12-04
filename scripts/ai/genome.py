import numpy as np
from scripts.ai.connection_gene import ConnectionGene
from scripts.ai.node import Node

class Genome:
    def __init__(self, inputs, outputs):
        self.genes = []  # List of connectionGene objects
        self.nodes = []  # List of Node objects
        self.inputs = inputs
        self.outputs = outputs
        self.layers = 2
        self.nextNode = 0
        self.biasNode = None
        self.network = []  # List of Node objects in order for the NN

        # Create input nodes
        for i in range(self.inputs):
            self.nodes.append(Node(i))
            self.nextNode += 1
            self.nodes[i].layer = 0

        # Create output nodes
        for i in range(self.outputs):
            self.nodes.append(Node(i + self.inputs))
            self.nodes[i + self.inputs].layer = 1
            self.nextNode += 1

        # Create bias node
        self.nodes.append(Node(self.nextNode))
        self.biasNode = self.nextNode
        self.nextNode += 1
        self.nodes[self.biasNode].layer = 0

    def get_node(self, node_number):
        for node in self.nodes:
            if node.number == node_number:
                return node
        return None

    def connect_nodes(self):
        for node in self.nodes:
            node.output_connections = []  # Clearing the list

        for gene in self.genes:
            gene.from_node.output_connections.append(gene)

    def feed_forward(self, input_values):
        for i in range(self.inputs):
            self.nodes[i].output_value = input_values[i]

        self.nodes[self.biasNode].output_value = 1  # Output of bias is 1

        for node in self.network:
            node.engage()

        outs = np.array([self.nodes[self.inputs + i].output_value for i in range(self.outputs)])

        for node in self.nodes:
            node.input_sum = 0

        return outs

    def generate_network(self):
        self.connect_nodes()
        self.network = []

        for l in range(self.layers):
            for node in self.nodes:
                if node.layer == l:
                    self.network.append(node)
                    
    def add_node(self, innovation_history):
        if len(self.genes) == 0:
            self.add_connection(innovation_history)
            return

        random_connection = np.random.randint(len(self.genes))
        while self.genes[random_connection].from_node == self.nodes[self.biasNode] and len(self.genes) != 1:
            random_connection = np.random.randint(len(self.genes))

        self.genes[random_connection].enabled = False

        new_node_no = self.nextNode
        self.nodes.append(Node(new_node_no))
        self.nextNode += 1

        # Add new connection with weight 1
        connection_innovation_number = self.get_innovation_number(innovation_history, self.genes[random_connection].from_node, self.get_node(new_node_no))
        self.genes.append(ConnectionGene(self.genes[random_connection].from_node, self.get_node(new_node_no), 1, connection_innovation_number))

        # Add another connection with the weight of the disabled connection
        connection_innovation_number = self.get_innovation_number(innovation_history, self.get_node(new_node_no), self.genes[random_connection].to_node)
        self.genes.append(ConnectionGene(self.get_node(new_node_no), self.genes[random_connection].to_node, self.genes[random_connection].weight, connection_innovation_number))
        
        self.get_node(new_node_no).layer = self.genes[random_connection].from_node.layer + 1

        # Increment layer numbers and adjust layers if needed
        if self.get_node(new_node_no).layer == self.genes[random_connection].to_node.layer:
            for i in range(len(self.nodes) - 1):
                if self.nodes[i].layer >= self.get_node(new_node_no).layer:
                    self.nodes[i].layer += 1
            self.layers += 1

        self.connect_nodes()

    def add_connection(self, innovation_history):
        if self.fully_connected():
            print("connection failed")
            return

        # Randomly select nodes for new connection
        random_node1 = np.random.randint(len(self.nodes))
        random_node2 = np.random.randint(len(self.nodes))
        while self.random_connection_nodes_are_shit(random_node1, random_node2):
            random_node1 = np.random.randint(len(self.nodes))
            random_node2 = np.random.randint(len(self.nodes))

        # Ensure correct order of nodes based on layers
        if self.nodes[random_node1].layer > self.nodes[random_node2].layer:
            random_node1, random_node2 = random_node2, random_node1

        connection_innovation_number = self.get_innovation_number(innovation_history, self.nodes[random_node1], self.nodes[random_node2])
        new_connection = ConnectionGene(self.nodes[random_node1], self.nodes[random_node2], np.random.uniform(-1, 1), connection_innovation_number)
        self.genes.append(new_connection)

        self.connect_nodes()

    def random_connection_nodes_are_shit(self, r1, r2):
        if self.nodes[r1].layer == self.nodes[r2].layer:
            return True
        if self.nodes[r1].is_connected_to(self.nodes[r2]):
            return True
        return False

    def get_innovation_number(self, innovation_history, from_node, to_node):
        # ... implementation here
        pass

    def fully_connected(self):
        # ... implementation here
        pass

    def mutate(self, innovation_history):
        # ... implementation here
        pass

    def crossover(self, parent2):
        # ... implementation here
        pass

    def matching_gene(self, parent2, innovation_number):
        # ... implementation here
        pass

    def print_genome(self):
        # ... implementation here
        pass

    def clone(self):
        # ... implementation here
        pass

    # Additional methods like draw_genome() if we are ready to visualize the training process