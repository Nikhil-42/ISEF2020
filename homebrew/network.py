import math
import random
import numpy as np
import numba
from numba import jitclass, jit, njit
from numba import int32, int64, float32, float64, uint8
import timeit

import time

# Activation Functions and Derivatives
a = 0.01

def gen_repeated_range(start, stop, repeats, step=1, collate=False):
    if not collate:
        for n in range(start, stop, step):
            for i in range(repeats):
                yield n
    else:
        for i in range(repeats):
            for n in range(start, stop, step):
                yield n    

@njit
def relu(x: float64) -> float64:
    return x if x > 0.0 else 0.0

@njit
def leaky_relu(x):
    return x if x > 0.0 else x * 0.01

@njit
def sigmoid(x: float64) -> float64:
    return 1/(1+math.exp(-x))

@njit
def d_relu(x: float64) -> float64:
    return 1.0 if x > 0.0 else 0.0

@njit
def d_leaky_relu(x):
    return 1 if x > 0.0 else 0.01

@njit
def d_sigmoid(x: float64) -> float64:
    return 1/(1+math.exp(-x))

network_spec = [
    ('input_shape', int32),
    ('output_shape', int32),
    ('node_count', int32),
    ('nodes', float64[:,:]),
    ('to', int32[:,:]),
    ('frm', int32[:,:]),
    ('deps', int32[:,:]),
    ('weights', float64[:,:]),
    ('connections', numba.typeof({(int32(0), int32(0))})),
    ('learning_rate', float32),
    ('id', int32)
]

@jitclass(network_spec)
class JIT_Network:

    def __init__(self, input_shape: int, output_shape: int, node_count=0, learning_rate=0.01, id_num=-1):
        # np.random.seed(id_num)
        # random.seed(id_num)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.node_count = input_shape + output_shape if node_count<input_shape+output_shape else node_count
        self.nodes = np.empty(shape=(self.node_count, 3), dtype=float64)
        self.nodes.fill(-1.0)
        self.to = np.empty(shape=(self.node_count, self.node_count), dtype=int32)
        self.frm = np.empty(shape=(self.node_count, self.node_count), dtype=int32)
        for i in range(self.node_count):
            self.to[i, -1] = 0
            self.frm[i, -1] = 0
        self.deps = np.column_stack((np.zeros(shape=self.node_count, dtype=int32), np.full(shape=self.node_count, fill_value=self.node_count, dtype=int32)))
        # Initalizing weights from -1 to 1 is more effective than setting them from 0 to 1
        self.weights = (np.random.random((self.node_count, self.node_count)).astype(float64)*2-1)/self.node_count
        self.connections = {(int32(0), int32(0))}
        self.learning_rate = learning_rate
        self.id = id_num
        # print("Initialized a JIT_Network: [Input: ", self.input_shape, ", Output: ", self.output_shape, ", Node Count :", self.node_count, "]")

    def get_activation(self, node_i: int32) -> float64:
        p_activation = 0.0

        for from_node_i in self.frm[node_i][:self.frm[node_i, self.node_count-1]]:
            p_activation += self.weights[node_i, from_node_i] * self.nodes[from_node_i, 0]

        #Apply bias then activation function
        self.nodes[node_i, 0] = leaky_relu(p_activation + self.nodes[node_i, 2])

        return self.nodes[node_i, 0]
    
    def get_backward_delta(self, node_i: int32) -> float64:
        p_error = 0.0

        for to_node_i in self.to[node_i][:self.to[node_i, self.node_count-1]]:
            p_error += self.weights[to_node_i, node_i] * self.nodes[to_node_i, 1]
        
        self.nodes[node_i, 1] = p_error * d_leaky_relu(self.nodes[node_i, 0])
            
        return self.nodes[node_i, 1]

    def forward_propagate(self, input_layer):
        # print("Beginning forward propagate with: ", input_layer)

        # Initialize the input layer
        for node_i in range(self.input_shape):
            self.nodes[node_i, 0] = input_layer[node_i]
        
        for node_i in range(self.input_shape, self.node_count):
            self.get_activation(node_i)

        # Get the activation of each output node and return the output_layer
        return self.nodes[-self.output_shape:, 0]

    def backward_propagate_error(self, output_deltas):

        # Initialize the output layer
        for node_i in range(self.output_shape):
            self.nodes[self.node_count-self.output_shape+node_i, 1] = output_deltas[node_i]

        # Get the activation of each input node and return the input_layer
        for node_i in range(self.node_count-self.output_shape-1, -1, -1):
            self.get_backward_delta(node_i)

    def update_weights(self):
        for connection in self.connections:
            self.weights[connection] += self.learning_rate * self.nodes[connection[0], 1] * self.nodes[connection[1], 0]
        for node_i in range(self.node_count):
            self.nodes[node_i, 2] += self.learning_rate * self.nodes[node_i, 1]

    def predict(self, inputs):
        if (0, 0) in self.connections:
            print("Network hasn't been trained")
            
        outputs = np.zeros((len(inputs), self.output_shape), dtype=float64)
        for i in numba.prange(len(inputs)):
            outputs[i] = self.forward_propagate(inputs[i])
        return outputs

    def validate(self, val_x, val_y, compare):
        if (0, 0) in self.connections:
            print("Network hasn't been trained")

        correct = 0
        for i in range(len(val_x)):
            output_layer = self.forward_propagate(val_x[i])
            if compare(output_layer, val_y[i]):
                correct += 1
            else:
                # print(output_layer, "=/=", val_y[i])
                pass
        return correct/len(val_x)

    def train(self, x: list, y: list, n_epoch=1, batch_size=1000, target_error=0.01):

        # # Generate forward layers
        # layer = [0]

        # forward_layers = []
        # for node_i in range(self.input_shape, self.node_count):
        #     if self.deps[node_i, 0] < layer[0]:
        #         layer.append(node_i)
        #     else:
        #         forward_layers.append(layer)
        #         layer = [node_i]
        # forward_layers.append(layer)
        # forward_layers = forward_layers[1:]

        # # Generate backward layers
        # layer = [self.node_count]

        # backward_layers = []
        # for node_i in range(self.node_count-self.output_shape-1, -1, -1):
        #     if self.deps[node_i, 1] > layer[0]:
        #         layer.append(node_i)
        #     else:
        #         backward_layers.append(layer)
        #         layer = [node_i]
        # backward_layers.append(layer)
        # backward_layers = backward_layers[1:]

        if batch_size > len(x):
            batch_size = len(x)

        if (0, 0) in self.connections:
            self.remove_connection((0, 0))

        self.nodes[:, 2] = np.random.random(self.node_count)
        
        acc_error = 0.0
        set_count = 0

        break_out = False

        for epoch in range(n_epoch):
            sum_error = 0.0
            # np.random.seed(12)

            for b in range(0, len(x), batch_size):
                error = 0.0
                for index in range(batch_size):
                    i = b + index
                    output_layer = self.forward_propagate(x[i])
                    self.backward_propagate_error(np.array([(y[i, j] - output_layer[j]) * d_leaky_relu(output_layer[j]) for j in range(self.output_shape)], dtype=float64))
                    self.update_weights()
                    for j in range(self.output_shape):
                        error += (y[i, j] - output_layer[j])**2

                sum_error += error
                set_count += batch_size
                # print("Batch:", int(b/batch_size), " AvgError:", error/batch_size)
                if np.isnan(error) or error/batch_size <= target_error:
                    # print("Error threshold reached.")
                    break_out = True
                    break
            # print("Epoch: ", epoch, " Summed Error: ", sum_error)
            acc_error += sum_error
            if break_out:
                break_out
        return acc_error

    def set_weight(self, connection, weight):
        self.weights[connection] = weight

    def get_weight(self, connection):
        return self.weights[connection]

    def add_connection(self, to: int32, frm: int32):
        if to <= frm or (to, frm) in self.connections or to < self.input_shape or frm > self.node_count-self.output_shape:
            return
        self.to[frm, self.to[frm, self.node_count-1]] = to
        self.to[frm, self.node_count-1] += 1
        self.deps[to, 0] = max((self.deps[to, 0], frm))
        self.frm[to, self.frm[to, self.node_count-1]] = frm
        self.frm[to, self.node_count-1] += 1
        self.deps[frm, 1] = min((to, self.deps[frm, 1]))
        self.connections.add((int32(to), int32(frm)))

    def add_connections(self, tos, frms):
        new_connections = set(map(lambda x, y: (x, y), tos, frms))
        new_connections.difference_update(self.connections)
        self.connections.update(new_connections)
        
        for connection in new_connections:
            to, frm = connection
            if to <= frm or to < self.input_shape or frm > self.node_count-self.output_shape:
                self.connections.remove(connection)
                continue
            # print("connecting node[", frm, "] to node[", to, "]")
            self.to[frm, self.to[frm, self.node_count-1]] = to
            self.to[frm, self.node_count-1] += 1
            self.deps[to, 0] = max((self.deps[to, 0], frm))
            self.frm[to, self.frm[to, self.node_count-1]] = frm
            self.frm[to, self.node_count-1] += 1
            self.deps[frm, 1] = min((to, self.deps[frm, 1]))
            # print("connected  node[", frm, "] to node[", to, "]")


    def set_connections(self, connections):
        self.to[:,-1].fill(0)
        self.frm[:,-1].fill(0)
        self.connections = connections
        
        for connection in connections:
            to, frm = connection
            if to <= frm or to < self.input_shape or frm > self.node_count-self.output_shape:
                self.connections.remove(connection)
                continue
            # print("connecting node[", frm, "] to node[", to, "]")
            self.to[frm, self.to[frm, self.node_count-1]] = to
            self.to[frm, self.node_count-1] += 1
            self.deps[to, 0] = max((self.deps[to, 0], frm))
            self.frm[to, self.frm[to, self.node_count-1]] = frm
            self.frm[to, self.node_count-1] += 1
            self.deps[frm, 1] = min((to, self.deps[frm, 1]))
            # print("connected  node[", frm, "] to node[", to, "]")
    
    def remove_connection(self, connection):
        if connection in self.connections:
            self.connections.remove(connection)   
        self.set_connections(self.connections)

    def remove_random_connection(self):
        if len(self.connections > 1):
            self.remove_connection(self.connections.pop())
    
@njit
def clone(in_network):
    network = JIT_Network(0, 0, 0, 0.0, -1)
    network.input_shape = in_network.input_shape
    network.output_shape = in_network.output_shape
    network.node_count = in_network.node_count
    network.nodes = in_network.nodes.copy()
    network.to = in_network.to.copy()
    network.frm = in_network.frm.copy()
    network.deps = in_network.deps.copy()
    network.weights = in_network.weights.copy()
    network.connections = in_network.connections.copy()
    network.learning_rate = in_network.learning_rate
    network.id = in_network.id
    return network


def run_xor_test():
    # np.random.seed(0)
    # random.seed(0)

    x = np.array([[1, 0] if rand < 0.25 else [0, 1] if rand < 0.5 else [1, 1] if rand < 0.75 else [0,0] for rand in np.random.random(60000)], dtype=int)
    y = np.array([[1,] if np.sum(itm)==1 else [0,] for itm in x], dtype=int)

    node_count = 5

    model = JIT_Network(input_shape=2, output_shape=1, node_count=node_count, learning_rate=0.1, id_num=10)
    # for i in range(2, node_count-1):
    #     model.add_connection(i, 0)
    #     model.add_connection(i, 1)
    #     model.add_connection(node_count-1, i)

    model.add_connections(np.array([2,2,3,3,4,4]), np.array([0, 1, 0, 1, 2, 3]))
    
    @njit
    def nopython_round(output_layer, expected):
        truth = 1
        for i, out in enumerate(output_layer):
            truth += not round(out) == expected[i]
        return truth == 1

    print("Training error: ", model.train(x, y, 1, 1000, 0.001))
    print("Validation accuracy: ", model.validate(x, y, nopython_round))
    print(model.predict(((1, 1), (1, 0), (0, 1), (0, 0))))

if __name__ == '__main__':
    # print(min(timeit.repeat(stmt='network.run_xor_test()', setup='gc.enable(); import network; ', repeat=10, number=1)))
    run_xor_test()

# 100000000 loops, best of 3: 0.00701 usec per loop
# 100000000 loops, best of 3: 0.00692 usec per loop
# 100000000 loops, best of 3: 0.00690 usec per loop
