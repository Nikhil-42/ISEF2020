import math
import random

# Activation Functions and Derivatives

def relu(x: float) -> float:
    return x if x > 0 else 0

def d_relu(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def leaky_relu(x: float) -> float:
    return x if x > 0 else 0.01*x

def d_leaky_relu(x: float) -> float:
    return 1.0 if x > 0 else 0.01

def sigmoid(x: float) -> float:
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x: float) -> float:
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
        self.connections = set()
        self.nodes = [{'to': set(), 'from': set(), 'output': None, 'delta': None, 'bias': None, 'activation-func': activation_func, 'activation-d-func': d_funcs[activation_func]} for node in range(node_count)]
        self.learning_rate = learning_rate
        print("Initialized a Network: [Input: ", input_shape, ", Output: ", output_shape, ", Node Count :", node_count, "]")

    def get_activation(self, node_i: int):
        node_i = normalize_index(node_i, self.node_count)

        node = self.nodes[node_i]
        if node['output']==None:
            activation = 0.0
            for i, connect in enumerate(self.connections):
                if connect[0]==node_i:
                    activation += self.weights[connect] * self.get_activation(connect[1])
            node['output'] = node['activation-func'](activation + node['bias'])
        return node['output']

    def get_backward_delta(self, node_i: int ):
        node_i = normalize_index(node_i, self.node_count)

        node = self.nodes[node_i]
        if node['delta']==None:
            error = 0.0
            for i, connect in enumerate(self.connections):
                if connect[1]==node_i:
                    error += self.weights[connect] * self.get_backward_delta(connect[0])
            node['delta'] = error * d_funcs[node['activation-func']](node['output'])
        return node['delta']

    def forward_propagate(self, input_layer):
        # Clear previous activations
        for node in self.nodes:
            node['output'] = None
            node['delta'] = None

        # Initialize the input layer
        for node_i, activation in enumerate(input_layer):
            self.nodes[node_i]['output'] = activation

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
        for connect in self.connections:
                self.weights[connect] += self.learning_rate * self.nodes[connect[0]]['delta'] * self.nodes[connect[1]]['output']
                for node in self.nodes:
                    if not node['delta']==None:
                        node['bias'] += self.learning_rate * node['delta']

    def predict(self, inputs):
        outputs = []
        for input_layer in inputs:
            outputs.append(self.forward_propagate(input_layer))

        return outputs

    def train(self, x, y, n_epoch=1):
        if not hasattr(self, "weights"):
            self.weights = {connection: random.random() for connection in self.connections}
            for node in self.nodes:
                node['bias'] = random.random()
        
        for epoch in range(n_epoch):
            for i, input_layer in enumerate(x):
                sum_error = 0.0
                output_layer = self.forward_propagate(input_layer)
                output_deltas = [(y[i][j] - output_layer[j]) * self.nodes[-self.output_shape:][j]['activation-d-func'](output_layer[j]) for j in range(self.output_shape)]
                print("Input: ", input_layer, " Expected: ", y[i], " Result: ", output_layer, " Deltas: ", output_deltas)
                self.backward_propagate_error(output_deltas)
                self.update_weights()
                sum_error += sum([(y[i][j] - output_layer[j])**2 for j in range(self.output_shape)])
            print("Input: ", x[:5], " Output: ", self.predict(x[:5]))
            print("Epoch: ", epoch, " Error: ", sum_error) 

    def add_connection(self, to: int, frm: int):
        to = normalize_index(to, self.node_count)
        frm = normalize_index(frm, self.node_count)
        # self.nodes[to]['from'].add(frm)
        # self.nodes[frm]['to'].add(to)
        self.connections.add((to, frm))



model = Network(input_shape=2, output_shape=1, node_count=5, learning_rate=0.01, activation_func=leaky_relu)
model.add_connection(-1, 3)
model.add_connection(-1, 2)
model.add_connection(2, 0)
model.add_connection(2, 1)
model.add_connection(3, 0)
model.add_connection(3, 1)
# model.add_connection(4, 2)
# model.add_connection(4, 3)
# model.add_connection(5, 2)
# model.add_connection(5, 3)

x = [(1, 0) if random.random() < 0.25 else (0, 1) if random.random() < 0.5 else (1, 1) if random.random() < 0.75 else (0,0) for num in range(1000)]
# x = [(1, 1) for i in range(100)]
y = [(1,) if sum(itm)==1 else (0,) for itm in x]

model.train(x, y, 50)

print([round(output[0]) for output in model.predict(((1, 1), (1, 0), (0, 1), (0, 0)))])
