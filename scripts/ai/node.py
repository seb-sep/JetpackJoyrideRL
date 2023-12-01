import numpy as np

class Node:
    def __init__(self, number):
        self.number = number
        self.input_sum = 0.0  # Current sum before activation
        self.output_value = 0.0  # Value after activation function is applied
        self.output_connections = []  # List of connectionGene objects
        self.layer = 0  # Layer of the node in the neural network

    def engage(self):
        # Apply the sigmoid function if not an input or bias node
        if self.layer != 0:
            self.output_value = self.sigmoid(self.input_sum)

        # Propagate the output value to the next nodes
        for connection in self.output_connections:
            if connection.enabled:
                connection.to_node.input_sum += connection.weight * self.output_value

    @staticmethod
    def sigmoid(x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-4.9 * x))
    
    @staticmethod
    def relu(x):
        # ReLU activation function
        return np.max(0, x)

    def is_connected_to(self, node):
        # Check if this node is connected to the given node
        if node.layer == self.layer:
            return False  # Nodes in the same layer cannot be connected

        if node.layer < self.layer:
            return any(conn.to_node == self for conn in node.output_connections)
        else:
            return any(conn.to_node == node for conn in self.output_connections)

    def clone(self):
        # Returns a copy of this node
        clone = Node(self.number)
        clone.layer = self.layer
        return clone
