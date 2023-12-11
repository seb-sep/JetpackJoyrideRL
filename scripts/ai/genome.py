import numpy as np
from scripts.ai.connection_gene import ConnectionGene
from scripts.ai.node import Node
import scipy.special as sp
from scripts.ai.connection_history import ConnectionHistory

class Genome:
    def __init__(self, inputs, outputs, crossover=False):
        self.genes = []  # List of connectionGene objects
        self.nodes = []  # List of Node objects
        self.inputs = inputs
        self.outputs = outputs
        self.layers = 2
        self.next_node = 0
        self.bias_node = None
        self.network = []  # List of Node objects in order for the NN

        # Create input nodes
        for i in range(self.inputs):
            self.nodes.append(Node(i))
            self.next_node += 1
            self.nodes[i].layer = 0

        # Create output nodes
        for i in range(self.outputs):
            self.nodes.append(Node(i + self.inputs))
            self.nodes[i + self.inputs].layer = 1
            self.next_node += 1

        # Create bias node
        self.nodes.append(Node(self.next_node))
        self.bias_node = self.next_node
        self.next_node += 1
        self.nodes[self.bias_node].layer = 0

        self.l1 = np.random.rand(8, 3)
        self.l2 = np.random.rand(3, 2)

    def get_node(self, node_number):
        for node in self.nodes:
            if node.number == node_number:
                return node
        return None

    def feed_forward(self, input_values):
        for i in range(self.inputs):
            # print(len(self.nodes))
            # print(len(input_values))
            self.nodes[i].output_value = input_values[i]

        self.nodes[self.bias_node].output_value = 1  # Output of bias is 1

        for node in self.network:
            node.engage()

        outs = np.array([self.nodes[self.inputs + i].output_value for i in range(self.outputs)])

        for node in self.nodes:
            node.input_sum = 0

        return np.argmax(outs)
    
    def feed_forward2(self, inputs):
        inputs = [1] + inputs # add bias
        inputs = np.array(inputs)

        x = sp.expit(sp.expit(inputs.dot(self.l1)).dot(self.l2))
        return np.argmax(x) # 0 for no lift, 1 for lift

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
        print('bias node: ', self.nodes[self.bias_node].number)
        [print(gene.from_node.number, gene.to_node.number) for gene in self.genes]
        while self.genes[random_connection].from_node == self.nodes[self.bias_node] and len(self.genes) != 1:
            # print('bias node: ', self.nodes[self.bias_node].number)
            # [print(gene.from_node.number, gene.to_node.number) for gene in self.genes]
            random_connection = np.random.randint(len(self.genes))
            # print('random_connection', random_connection)

        self.genes[random_connection].enabled = False

        new_node_no = self.next_node
        self.nodes.append(Node(new_node_no))
        self.next_node += 1

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

    def add_connection(self, innovation_history: ConnectionHistory):
        if self.fully_connected():
            print("connection failed")
            return

        # Randomly select nodes for new connection
        random_node1 = np.random.randint(len(self.nodes))
        random_node2 = np.random.randint(len(self.nodes))
        while self.check_bad_connections(random_node1, random_node2):
            random_node1 = np.random.randint(len(self.nodes))
            random_node2 = np.random.randint(len(self.nodes))

        # Ensure correct order of nodes based on layers
        if self.nodes[random_node1].layer > self.nodes[random_node2].layer:
            random_node1, random_node2 = random_node2, random_node1

        connection_innovation_number = self.get_innovation_number(innovation_history, self.nodes[random_node1], self.nodes[random_node2])
        new_connection = ConnectionGene(self.nodes[random_node1], self.nodes[random_node2], np.random.uniform(-1, 1), connection_innovation_number)
        self.genes.append(new_connection)

        self.connect_nodes()

    def check_bad_connections(self, r1: Node, r2: Node):
        if self.nodes[r1].layer == self.nodes[r2].layer:
            return True
        if self.nodes[r1].is_connected_to(self.nodes[r2]):
            return True
        return False

    # returns the innovation number for the new mutation
    # if this mutation has never been seen before then it will be given a new unique innovation number
    def get_innovation_number(self, innovation_history, from_node, to_node):
        is_new = True
        connection_innovation_no = 1000
        for innovation in innovation_history:
            if innovation.matches(self, from_node, to_node):
                is_new = False
                connection_innovation_no = innovation.innovation_no
                break

        if is_new:
            inno_nums = []
            [inno_nums.append(gene.innovation_no) for gene in self.genes]
            innovation_history.append(ConnectionHistory(from_node, to_node, connection_innovation_no, inno_nums)) 
        
        return connection_innovation_no

    def fully_connected(self):
        # ... implementation here
        pass

    def mutate(self, innovation_history):
        if len(self.genes) == 0:  
            print("adding connection")
            self.add_connection(innovation_history) 
            
        if np.random.rand() < 0.8: 
            print("mutating weight")
            [gene.mutate_weight() for gene in self.genes]
            
        if np.random.rand() < 0.10: 
            print("adding connection")
            self.add_connection(innovation_history)
            
        if np.random.rand() < 0.01: 
            print("adding node")
            self.add_node(innovation_history)
        
    def crossover(self, parent2):
        child = Genome(self.inputs, self.outputs, True)
        child.genes.clear()
        child.nodes.clear()
        child.layers = self.layers
        child.next_node = self.next_node
        child.bias_node = self.bias_node
        child_genes = []
        is_enabled = []
        for gene in self.genes:
            set_enabled= True
            parent_2_gene = self.matching_gene(parent2, gene.innovation_no)
            if parent_2_gene != -1:
                if not gene.enabled or not parent2.genes[parent_2_gene].enabled:
                    if np.random.random() < 0.75:
                        set_enabled = False
                if np.random.random() < 0.5:
                    print(f'adding gene {gene.from_node.number} {gene.to_node.number}')
                    if gene.from_node.number is not self.bias_node: child_genes.append(gene)
                else:
                    print(f'adding gene {parent2.genes[parent_2_gene].from_node.number} {parent2.genes[parent_2_gene].to_node.number}')
                    if parent2.genes[parent_2_gene].from_node.number is not parent2.bias_node: child_genes.append(parent2.genes[parent_2_gene])
            else:
                child_genes.append(gene)
                set_enabled = gene.enabled
            is_enabled.append(set_enabled)
        for node in self.nodes:
            child.nodes.append(node.clone())
        for i, gene in enumerate(child_genes):
            from_node_clone = child.get_node(gene.from_node.number)
            to_node_clone = child.get_node(gene.to_node.number)
            cloned_gene = gene.clone(from_node_clone, to_node_clone)
            cloned_gene.enabled = is_enabled[i]
            print(f'adding gene {cloned_gene.from_node.number} {cloned_gene.to_node.number}')
            if cloned_gene.from_node.number is not child.bias_node: child.genes.append(cloned_gene)
        child.connect_nodes()
        return child

    def matching_gene(self, other_genome, innovation_number):
        for i in range(len(other_genome.genes)):
            if other_genome.genes[i].innovation_no == innovation_number:
                return i
        return -1

    # prints the genome network in console
    def print_genome(self):
        # ... implementation here
        
        pass

    def clone(self):
        clone = Genome(self.inputs, self.outputs)

        # Copy nodes
        for node in self.nodes:
            clone.nodes.append(node.clone())

        # Copy genes
        for gene in self.genes:
            from_node_clone = clone.get_node(gene.from_node.number)
            to_node_clone = clone.get_node(gene.to_node.number)
            clone.genes.append(gene.clone(from_node_clone, to_node_clone))

        clone.layers = self.layers
        clone.next_node = self.next_node
        clone.bias_node = self.bias_node
        clone.connect_nodes()

        return clone

    def connect_nodes(self):
        # print('num nodes', len(self.nodes))
        [node.output_connections.clear() for node in self.nodes]
        # print('num genes', len(self.genes))
        [gene.from_node.output_connections.append(gene) for gene in self.genes]

