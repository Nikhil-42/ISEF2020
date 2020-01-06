import math
import random
import numpy as np
import numba
from numba import jitclass, jit, njit
from numba import int32, int64, float32, float64, uint8

# Activation Functions and Derivatives
a = 0.01

@njit
def relu(x: float64) -> float64:
    return x if x > 0.0 else a*x

@njit
def d_relu(x: float64) -> float64:
    return 1.0 if x > 0.0 else a

np.random.seed(0)
random.seed(0)

network_spec = [
    ('input_shape', int32),
    ('output_shape', int32),
    ('node_count', int32),
    ('nodes', float64[:,:]),
    ('to', int32[:,:]),
    ('frm', int32[:,:]),
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
        self.to.fill(-1)
        self.frm = np.empty(shape=(node_count, node_count), dtype=int32)
        self.frm.fill(-1)
        for i in range(node_count):
            self.to[i, -1] = 0
            self.frm[i, -1] = 0
        self.weights = numba.typed.Dict.empty(key_type=int64, value_type=float64)
        self.activation_funcs = np.array((node_count,), dtype=uint8)
        self.learning_rate = learning_rate
        print("Initialized a JIT_Network: [Input: ", input_shape, ", Output: ", output_shape, ", Node Count :", node_count, "]")

    def get_activation(self, node_i: int32) -> float64:
        current_i = node_i
        p_activation = 0.0
        stack = []
        used = 0
        
        while(True):
            # print("Calculating activation of Node #", current_i, " from: ", self.frm[current_i][used:self.frm[current_i, self.node_count-1]])
            for from_node_i in self.frm[current_i][used:self.frm[current_i, self.node_count-1]]:
                if from_node_i == -1:
                    break
                if self.nodes[from_node_i, 0] == -1.0:
                    stack.append((current_i, p_activation, used+1))
                    current_i = from_node_i
                    p_activation = 0.0
                    used = 0
                    break
                else:
                    p_activation += self.weights[current_i << 32 | from_node_i] * self.nodes[from_node_i, 0]
                    used += 1
            else:
                #Apply bias then activation function
                
                self.nodes[current_i, 0] = relu(p_activation + self.nodes[current_i, 2])
                if current_i==node_i:
                    break
                else:
                    temp_i, p_activation, used = stack.pop()
                    p_activation += self.weights[temp_i << 32|current_i] * self.nodes[current_i, 0]
                    current_i = temp_i
                #print("Node #", current_i, " = ", output)
        return self.nodes[node_i, 0]
    
    def get_backward_delta(self, node_i: int32) -> float64:
        current_i = node_i
        p_error = 0.0
        stack = []
        used = 0
        
        while(True):
            # print("Evaluating Node #", current_i, " from ", self.to[current_i][used:self.to[current_i, self.node_count-1]])

            for to_node_i in self.to[current_i][used:self.to[current_i, self.node_count-1]]:
                if to_node_i == -1:
                    break
                if self.nodes[to_node_i, 1] == -1.0:
                    stack.append((current_i, p_error, used+1))
                    current_i = to_node_i
                    p_error = 0.0
                    used = 0
                    break
                else:
                    p_error += self.weights[to_node_i << 32 | current_i] * self.nodes[to_node_i, 1]
                    used += 1
            else:
                self.nodes[current_i, 1] = p_error * d_relu(self.nodes[current_i, 0])
                if current_i==node_i:
                    break
                else:
                    temp_i, p_error, used = stack.pop()
                    p_error += self.weights[current_i << 32| temp_i] * self.nodes[current_i, 1]
                    current_i = temp_i
        return self.nodes[node_i, 1]

    def forward_propagate(self, input_layer):
        # print("Beginning forward propagate with: ", input_layer)

        # Clear previous activations
        for node_i in range(self.node_count):
            self.nodes[node_i, 0] = -1.0

        # Initialize the input layer
        for node_i in range(self.input_shape):
            self.nodes[node_i, 0] = input_layer[node_i]

        # Get the activation of each output node and return the output_layer
        return np.array([self.get_activation(self.node_count-self.output_shape+num) for num in range(self.output_shape)], dtype=float64)

    def backward_propagate_error(self, output_deltas):
        # Clear previous activations
        for node_i in range(self.node_count):
            self.nodes[node_i, 1] = -1.0

        # Initialize the output layer
        for node_i in range(self.output_shape):
            self.nodes[self.node_count-self.output_shape+node_i, 1] = output_deltas[node_i]

        # Get the activation of each input node and return the input_layer
        for node_i in range(self.input_shape):
            self.get_backward_delta(node_i)

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

    def train(self, x: list, y: list, n_epoch=1):
        
        np.random.seed(0)
        for i, rand in enumerate(np.random.random(self.node_count)):
            self.nodes[i, 2] = rand
        
        for epoch in range(n_epoch):
            for i in range(len(x)):
                input_layer = x[i]
                expected_layer = y[i]
                sum_error = 0.0
                output_layer = self.forward_propagate(input_layer)
                output_gradients = [d_relu(output_layer[j]) for j in range(self.output_shape)]
                output_deltas = np.array([(expected_layer[j] - output_layer[j]) * output_gradients[j] for j in range(self.output_shape)], dtype=float64)
                self.backward_propagate_error(output_deltas)
                self.update_weights()
                for j in range(self.output_shape):
                    sum_error += (expected_layer[j] - output_layer[j])**2
            print("Epoch: ", epoch, " Error: ", sum_error)

    def add_connection(self, to: int32, frm: int32):
        self.to[frm, self.to[frm, self.node_count-1]] = to
        self.to[frm, self.node_count-1] += 1
        self.frm[to, self.frm[to, self.node_count-1]] = frm
        self.frm[to, self.node_count-1] += 1
        self.weights[(to << 32) | frm] = random.random()
        # print("Added connection [", to,", ", frm, "] with Weight: ", self.weights[(to << 32) | frm])
                
def run_xor_test():
    model = JIT_Network(input_shape=2, output_shape=1, node_count=103, learning_rate=0.01)
    # model.add_connections(np.array(range(2, 102)), np.array(range(0,2)))
    # model.add_connections(np.array([102,]), np.array(range(2, 102))
    for i in range(2, 102):
        model.add_connection(i, 0)
        model.add_connection(i, 1)
        model.add_connection(102, i)

    x = np.array([[1, 0] if rand < 0.25 else [0, 1] if rand < 0.5 else [1, 1] if rand < 0.75 else [0,0] for rand in np.random.random(60000)], dtype=int)
    y = np.array([[1,] if np.sum(itm)==1 else [0,] for itm in x], dtype=int)

    model.train(x, y, 2)

    print([round(output[0]) for output in model.predict(((1, 1), (1, 0), (0, 1), (0, 0)))])

if __name__ == '__main__':
    run_xor_test()
    #print(timeit.timeit(stmt='network.run_xor_test()', setup='gc.enable(); import network; ', number=20))

# 231.3515047 : iterative
# 193.9866752 : recursive
# 1339.0884785 : with @jit