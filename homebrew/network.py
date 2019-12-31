import math
from decimal import Decimal
import random
import numpy as np

# Activation Functions and Derivatives

a = Decimal(0.01)
D_one = Decimal(1)
D_zero = Decimal(0)

random.seed(12)

def relu(x):
    return x if x > 0 else D_zero

def d_relu(x):
    return D_one if x >= 0 else D_zero

def leaky_relu(x):
    return x if x > 0 else a*x

def d_leaky_relu(x):
    return D_one if x >= 0 else a

def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + Decimal.exp(x))
    return 1 / (1 + Decimal.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

d_funcs = {relu: d_relu, leaky_relu: d_leaky_relu, sigmoid: d_sigmoid}

def normalize_index(index: int, length: int):
    if index < 0:
        return index + length
    return index

class Network:

    def __init__(self, input_shape: int, output_shape: int, node_count=0, activation_func=leaky_relu, learning_rate=0.1):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.node_count = input_shape + output_shape if node_count<input_shape+output_shape else node_count
        self.nodes = [{'to': set(), 'from': set(), 'output': None, 'delta': None, 'bias': None, 'activation-func': activation_func, 'activation-d-func': d_funcs[activation_func]} for node in range(node_count)]
        self.weights = dict()
        self.learning_rate = Decimal(learning_rate)
        print("Initialized a Network: [Input: ", input_shape, ", Output: ", output_shape, ", Node Count :", node_count, "]")

    def get_activation(self, node_i: int) -> Decimal:
        node_i = normalize_index(node_i, self.node_count)
        node = self.nodes[node_i]

        if node['output']==None:
            # Grab the activations of all input nodes simultaneously
            
            # Calculate weighted sum
            activation = Decimal(0)
            for from_node_i in node['from']:
                activation += self.weights[(node_i, from_node_i)] * self.get_activation(from_node_i)

            # Apply bias then activation function
            node['output'] = node['activation-func'](activation + node['bias'])

        return node['output']

    def get_backward_delta(self, node_i: int ) -> Decimal:
        node_i = normalize_index(node_i, self.node_count)
        node = self.nodes[node_i]

        if node['delta']==None:

            # Calculate weighted sum
            error = Decimal(0)
            for to_node_i in node['to']:
                error += self.weights[(to_node_i, node_i)] * self.get_backward_delta(to_node_i)


            node['delta'] = error * node['activation-d-func'](node['output'])

        return node['delta']

    def forward_propagate(self, input_layer):
        # Clear previous activations
        for node in self.nodes:
            node['output'] = None
            node['delta'] = None

        # Initialize the input layer
        for node_i, activation in enumerate(input_layer):
            self.nodes[node_i]['output'] = Decimal(activation)

        # Get the activation of each output node and return the output_layer
        return [self.get_activation(num-self.output_shape) for num in range(self.output_shape)]

    def backward_propagate_error(self, output_deltas):
        # Clear previous activations
        for node in self.nodes:
            node['delta'] = None

        # Initialize the output layer
        for i, delta in enumerate(output_deltas):
            self.nodes[-1-i]['delta'] = delta

        # Get the activation of each input node and return the input_layer
        for num in range(self.input_shape):
            self.get_backward_delta(num)

    def update_weights(self):
        for connect in self.weights.keys():
            self.weights[connect] += self.learning_rate * self.nodes[connect[0]]['delta'] * self.nodes[connect[1]]['output']
        for node in self.nodes:
            if not node['delta']==None:
                node['bias'] += self.learning_rate * node['delta']

    def predict(self, inputs):
        outputs = []
        for input_layer in inputs:
            outputs.append(self.forward_propagate(input_layer))

        return outputs

    def train(self, x: list, y: list, n_epoch=1):
        for node in self.nodes:
            node['bias'] = Decimal(random.random())
        
        for epoch in range(n_epoch):
            for i, input_layer in enumerate(x):
                sum_error = Decimal(0.0)
                output_layer = self.forward_propagate(input_layer)
                output_deltas = [(Decimal(y[i][j]) - output_layer[j]) * self.nodes[-self.output_shape:][j]['activation-d-func'](output_layer[j]) for j in range(self.output_shape)]
                self.backward_propagate_error(output_deltas)
                self.update_weights()
                sum_error += sum([(y[i][j] - output_layer[j])**2 for j in range(self.output_shape)])
            print("Epoch: ", epoch, " Error: ", sum_error) 

    def add_connection(self, to: int, frm: int):
        to = normalize_index(to, self.node_count)
        frm = normalize_index(frm, self.node_count)
        self.nodes[to]['from'].add(frm)
        self.nodes[frm]['to'].add(to)
        self.weights[(to, frm)] = Decimal(random.random())

    def add_connections(self, to: list, frm: list):
        to = set([normalize_index(e, self.node_count) for e in to])
        frm = set([normalize_index(e, self.node_count) for e in frm])
        for node_i in to:
            (self.nodes[node_i]['from']).update(frm)
        for node_i in frm:
            (self.nodes[node_i]['to']).update(to)
        for to_i in to:
            for frm_i in frm:
                self.weights[(to_i, frm_i)] = Decimal(random.random())

if __name__ == "__main__":
    model = Network(input_shape=2, output_shape=1, node_count=103, learning_rate=0.01, activation_func=leaky_relu)
    model.add_connections(range(2, 102), range(0,2))
    model.add_connections((-1,), range(2, 102))

    x = [(1, 0) if random.random() < 0.25 else (0, 1) if random.random() < 0.5 else (1, 1) if random.random() < 0.75 else (0,0) for num in range(1000)]
    y = [(1,) if sum(itm)==1 else (0,) for itm in x]

    model.train(x, y, 50)

    print([round(output[0]) for output in model.predict(((1, 1), (1, 0), (0, 1), (0, 0)))])