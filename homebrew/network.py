import math
import random
import numpy as np
import pyopencl as cl
import numba
from numba import jitclass, jit, njit
from numba import int32, int64, float32, float64, uint8
import timeit

# Activation Functions and Derivatives
a = 0.01

@njit
def relu(x: float64) -> float64:
    return x if x > 0.0 else a*x

@njit
def d_relu(x: float64) -> float64:
    return 1.0 if x > 0.0 else a

network_spec = [
    ('input_shape', int32),
    ('output_shape', int32),
    ('node_count', int32),
    ('nodes', float64[:,:]),
    ('to', int32[:,:]),
    ('frm', int32[:,:]),
    ('deps', int32[:,:]),
    ('weights', numba.typeof(numba.typed.Dict.empty(key_type=int64, value_type=float64))),
    ('activation_funcs', uint8[:]),
    ('learning_rate', float32)
]

@jitclass(network_spec)
class JIT_Network:

    def __init__(self, input_shape: int, output_shape: int, node_count=0, learning_rate=0.1):
        np.random.seed(12)
        random.seed(12)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.node_count = input_shape + output_shape if node_count<input_shape+output_shape else node_count
        self.nodes = np.empty(shape=(node_count, 3), dtype=float64)
        self.nodes.fill(-1.0)
        self.to = np.empty(shape=(node_count, node_count), dtype=int32)
        self.frm = np.empty(shape=(node_count, node_count), dtype=int32)
        for i in range(node_count):
            self.to[i, -1] = 0
            self.frm[i, -1] = 0
        self.deps = np.column_stack((np.zeros(shape=node_count, dtype=int32), np.full(shape=node_count, fill_value=node_count, dtype=int32)))
        self.weights = numba.typed.Dict.empty(key_type=int64, value_type=float64)
        self.activation_funcs = np.array((node_count,), dtype=uint8)
        self.learning_rate = learning_rate
        print("Initialized a JIT_Network: [Input: ", input_shape, ", Output: ", output_shape, ", Node Count :", node_count, "]")

    def get_activation(self, node_i: int32) -> float64:
        p_activation = 0.0

        for from_node_i in self.frm[node_i][:self.frm[node_i, self.node_count-1]]:
            p_activation += self.weights[node_i << 32 | from_node_i] * self.nodes[from_node_i, 0]
        
        #Apply bias then activation function
        self.nodes[node_i, 0] = relu(p_activation + self.nodes[node_i, 2])

        return self.nodes[node_i, 0]
    
    def get_backward_delta(self, node_i: int32) -> float64:
        p_error = 0.0

        for to_node_i in self.to[node_i][:self.to[node_i, self.node_count-1]]:
            p_error += self.weights[to_node_i << 32 | node_i] * self.nodes[to_node_i, 1]
        
        self.nodes[node_i, 1] = p_error * d_relu(self.nodes[node_i, 0])
            
        return self.nodes[node_i, 1]

    def forward_propagate(self, input_layer):
        # print("Beginning forward propagate with: ", input_layer)

        # Clear previous activations
        for node_i in range(self.node_count):
            self.nodes[node_i, 0] = -1.0

        # Initialize the input layer
        for node_i in range(self.input_shape):
            self.nodes[node_i, 0] = input_layer[node_i]

        for node_i in range(self.input_shape, self.node_count):
            self.get_activation(node_i)

        # Get the activation of each output node and return the output_layer
        return np.array([self.nodes[i, 0] for i in range(self.node_count-self.output_shape, self.node_count)], dtype=float64)

    def backward_propagate_error(self, output_deltas):
        # Clear previous activations
        for node_i in range(self.node_count):
            self.nodes[node_i, 1] = -1.0

        # Initialize the output layer
        for node_i in range(self.output_shape):
            self.nodes[self.node_count-self.output_shape+node_i, 1] = output_deltas[node_i]

        # Get the activation of each input node and return the input_layer
        for node_i in range(self.node_count-self.output_shape-1, -1, -1):
            self.get_backward_delta(node_i)
        # print(self.nodes)

    def update_weights(self):
        for connection in self.weights.keys():
            self.weights[connection] += self.learning_rate * self.nodes[connection >> 32 & 0xFFFFFFFF][1] * self.nodes[connection & 0xFFFFFFFF][0]
        for node_i in range(self.node_count):
            if not self.nodes[node_i, 1]==-1.0:
                self.nodes[node_i, 2] += self.learning_rate * self.nodes[node_i, 1]

    def predict(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.forward_propagate(inputs[i]))

        return outputs

    def train(self, x: list, y: list, n_epoch=1, batch_size=1000, target_error=0.001):
        
        # Generate forward layers
        layer = [0]

        forward_layers = []
        for node_i in range(self.input_shape, self.node_count):
            if self.deps[node_i, 0] < layer[0]:
                layer.append(node_i)
            else:
                forward_layers.append(layer)
                layer = [node_i]
        forward_layers.append(layer)
        forward_layers = forward_layers[1:]

        # Generate backward layers
        layer = [self.node_count]

        backward_layers = []
        for node_i in range(self.node_count-self.output_shape-1, -1, -1):
            if self.deps[node_i, 1] > layer[0]:
                layer.append(node_i)
            else:
                backward_layers.append(layer)
                layer = [node_i]
        backward_layers.append(layer)
        backward_layers = backward_layers[1:]

        if batch_size > len(x):
            batch_size = len(x)
        
        np.random.seed(0)
        for i, rand in enumerate(np.random.random(self.node_count)):
            self.nodes[i, 2] = rand
        
        for epoch in range(n_epoch):
            sum_error = 0.0
            for b in range(0, len(x), batch_size):
                error = 0.0
                for index in range(batch_size):
                    i = b + index
                    output_layer = self.forward_propagate(x[i])
                    self.backward_propagate_error(np.array([(y[i, j] - output_layer[j]) * d_relu(output_layer[j]) for j in range(self.output_shape)], dtype=float64))
                    self.update_weights()
                    for j in range(self.output_shape):
                        error += (y[i, j] - output_layer[j])**2
                if error/batch_size <= target_error:
                    print("Error threshold reached.")
                    return
                sum_error += error
            print("Epoch: ", epoch, " Error: ", sum_error)

    def add_connection(self, to: int32, frm: int32):
        if to <= frm:
            return
        self.to[frm, self.to[frm, self.node_count-1]] = to
        self.to[frm, self.node_count-1] += 1
        self.deps[to, 0] = max((self.deps[to, 0], frm))
        self.frm[to, self.frm[to, self.node_count-1]] = frm
        self.frm[to, self.node_count-1] += 1
        self.deps[frm, 1] = min((to, self.deps[frm, 1]))
        self.weights[(to << 32) | frm] = random.random()
        # print("Added connection [", to,", ", frm, "] with Weight: ", self.weights[(to << 32) | frm])
                
def run_xor_test():
    np.random.seed(0)
    random.seed(0)

    x = np.array([[1, 0] if rand < 0.25 else [0, 1] if rand < 0.5 else [1, 1] if rand < 0.75 else [0,0] for rand in np.random.random(60000)], dtype=int)
    y = np.array([[1,] if np.sum(itm)==1 else [0,] for itm in x], dtype=int)

    model = JIT_Network(input_shape=2, output_shape=1, node_count=103, learning_rate=0.01)
    for i in range(2, 102):
        model.add_connection(i, 0)
        model.add_connection(i, 1)
        model.add_connection(102, i)

    model.train(x, y, 2)

    print([round(output[0]) for output in model.predict(((1, 1), (1, 0), (0, 1), (0, 0)))])

if __name__ == '__main__':
    # print(min(timeit.repeat(stmt='network.run_xor_test()', setup='gc.enable(); import network; ', repeat=10, number=1)))
    run_xor_test()

# 100000000 loops, best of 3: 0.00701 usec per loop
# 100000000 loops, best of 3: 0.00692 usec per loop
# 100000000 loops, best of 3: 0.00690 usec per loop
