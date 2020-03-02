import utils
import numba
import numpy as np
from numba import jitclass, jit, njit
from numba import int32, float32, float64, uint8

# Activations (take in a scalar)
a = 0.01

@njit
def relu(x):
    return x if x > 0.0 else 0.0

@njit
def leaky_relu(x):
    return x if x > 0.0 else x * a

@njit
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivatives
@njit
def d_relu(x):
    return 1.0 if x > 0.0 else 0.0

@njit
def d_leaky_relu(x):
    return 1.0 if x > 0.0 else a

@njit
def d_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

# Losses (take in an ndarray)
@njit
def mean_squared_error(y, yhat):
    return 0.5 * (y-yhat)**2

@njit
def binary_cross_entropy(y, yhat):
    return -(yhat*np.log(y) + (1-yhat)*np.log(1-y))

# Derivatives (take in ndarray)
@njit
def d_mean_squared_error(y, yhat):
    return y-yhat

@njit
def d_binary_cross_entropy(y, yhat):
    return -yhat/y + (1-yhat)/(1-y)

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
    ('activation_functions', uint8[:]),
    ('loss_i', uint8),
    ('learning_rate', float64),
    ('id', int32)
]

RAW = 0
OUT = 1
GRAD = 2
BIAS = 3

@jitclass(network_spec)
class JIT_Network:

    def __init__(self, input_shape: int, output_shape: int, node_count=0, learning_rate=0.01, activation_functions=None, loss_i=0, id_num=-1):
        np.random.seed(14)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.node_count = input_shape + output_shape if node_count<input_shape+output_shape else node_count
        self.nodes = np.empty(shape=(self.node_count, 4), dtype=float64)
        self.nodes.fill(-1.0)
        self.to = np.empty(shape=(self.node_count, self.node_count), dtype=int32)
        self.frm = np.empty(shape=(self.node_count, self.node_count), dtype=int32)
        for i in range(self.node_count):
            self.to[i, -1] = 0
            self.frm[i, -1] = 0
        self.deps = np.column_stack((np.zeros(shape=self.node_count, dtype=int32), np.full(shape=self.node_count, fill_value=self.node_count, dtype=int32)))
        # Initializing weights from -1 to 1 is more effective than setting them from 0 to 1
        self.weights = (np.random.random((self.node_count, self.node_count)).astype(float64)*2-1)/self.node_count
        self.connections = {(int32(0), int32(0))}
        if activation_functions==None:
            self.activation_functions = np.ones(node_count, dtype=uint8)
        else:
            self.activation_functions = activation_functions
        self.loss_i = loss_i
        self.learning_rate = learning_rate
        self.id = id_num
        
    def activation(self, i, x):
        if i == 0:
            return relu(x)
        elif i == 1:
            return leaky_relu(x)
        elif i == 2:
            return sigmoid(x)
    
    def d_activation(self, i, x):
        if i == 0:
            return d_relu(x)
        elif i == 1:
            return d_leaky_relu(x)
        elif i == 2:
            return d_sigmoid(x)

    def loss(self, i, y, yhat):
        if i == 0:
            return mean_squared_error(y, yhat)
        elif i == 1:
            return mean_squared_error(y, yhat)
        elif i == 2:
            return binary_cross_entropy(y, yhat)

    def d_loss(self, i, y, yhat):
        if i == 0:
            return d_mean_squared_error(y, yhat)
        elif i == 1:
            return d_mean_squared_error(y, yhat)
        elif i == 2:
            return d_binary_cross_entropy(y, yhat)
    
    def get_activation(self, node_i):
        p_activation = 0.0

        for from_node_i in self.frm[node_i][:self.frm[node_i, self.node_count-1]]:
            p_activation += self.weights[node_i, from_node_i] * self.nodes[from_node_i, OUT]

        # Apply bias
        self.nodes[node_i, RAW] = p_activation + self.nodes[node_i, BIAS]
        # then activation function
        self.nodes[node_i, OUT] = self.activation(self.activation_functions[node_i], self.nodes[node_i, RAW])

        return self.nodes[node_i, OUT]

    def get_backward_delta(self, node_i):
        p_error = 0.0

        for to_node_i in self.to[node_i][:self.to[node_i, self.node_count-1]]:
            p_error += self.weights[to_node_i, node_i] * self.nodes[to_node_i, GRAD]
        
        self.nodes[node_i, GRAD] = p_error * self.d_activation(self.activation_functions[node_i], self.nodes[node_i, RAW])
            
        return self.nodes[node_i, GRAD]

    def forward_propagate(self, input_layer):

        # Initialize the input layer
        for node_i in range(self.input_shape):
            self.nodes[node_i, OUT] = input_layer[node_i]
        
        for node_i in range(self.input_shape, self.node_count):
            self.get_activation(node_i)

        # Get the activation of each output node and return the output_layer
        return self.nodes[-self.output_shape:, OUT]

    def backward_propagate(self, expected_output):

        # Initialize the output layer
        d_out = np.empty(self.output_shape, dtype=float64)
        for i in range(self.output_shape):
            d_out[i] = self.d_activation(self.activation_functions[i], self.nodes[-self.output_shape+i, RAW])
        d_loss = self.d_loss(self.loss_i, self.nodes[-self.output_shape, OUT], expected_output)
        self.nodes[-self.output_shape:, GRAD] = d_loss * d_out

        # Get the activation of each input node and return the input_layer
        for node_i in range(self.node_count-self.output_shape-1, -1, -1):
            self.get_backward_delta(node_i)

        return self.loss(self.loss_i, self.nodes[-self.output_shape:, OUT], expected_output)

    def update_weights(self):
        for connection in self.connections:
            self.weights[connection] += self.learning_rate * -self.nodes[connection[0], GRAD] * self.nodes[connection[1], OUT]
        for node_i in range(self.node_count):
            self.nodes[node_i, BIAS] += self.learning_rate * -self.nodes[node_i, GRAD]

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
            
        return correct/len(val_x)

    def train(self, x: list, y: list, n_epoch=1, batch_size=1000, target_loss=0.01):

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

        self.nodes[:, BIAS] = np.random.random(self.node_count)
    
        set_count = 0

        break_out = False

        for epoch in range(n_epoch):
            for b in range(0, len(x), batch_size):
                error = 0.0
                for index in range(batch_size):
                    i = b + index

                    self.forward_propagate(x[i])
                    loss = self.backward_propagate(y[i])

                    self.update_weights()
                set_count += batch_size
            #     if np.isnan(error) or error/batch_size <= target_loss:
            #         break_out = True
            #         break
            # if break_out:
            #     break
        return loss

    def set_weight(self, connection, weight):
        self.weights[connection] = weight

    def get_weight(self, connection):
        return self.weights[connection]

    def setup_deps(self, to, frm):
        if to <= frm or to < self.input_shape or frm >= self.node_count-self.output_shape:
            return False
        self.to[frm, self.to[frm, self.node_count-1]] = to
        self.to[frm, self.node_count-1] += 1
        self.deps[to, 0] = max((self.deps[to, 0], frm))
        self.frm[to, self.frm[to, self.node_count-1]] = frm
        self.frm[to, self.node_count-1] += 1
        self.deps[frm, 1] = min((to, self.deps[frm, 1]))
        return True

    def add_connection(self, to: int32, frm: int32):
        if not (to, frm) in self.connections and self.setup_deps(to, frm):
            self.connections.add((int32(to), int32(frm)))

    def add_connections(self, tos, frms):
        new_connections = set(zip(tos, frms))
        new_connections.difference_update(self.connections)
        self.connections.update(new_connections)
        
        for connection in new_connections:
            if not self.setup_deps(*connection):
                self.connections.remove(connection)

    def set_connections(self, connections):
        self.to[:,-1].fill(0)
        self.frm[:,-1].fill(0)
        self.connections = connections
        
        for connection in connections:
            if not self.setup_deps(*connection):
                self.connections.remove(connection)
    
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

    np.random.seed(1)

    x = np.array([[1, 0] if rand < 0.25 else [0, 1] if rand < 0.5 else [1, 1] if rand < 0.75 else [0,0] for rand in np.random.random(60000)], dtype=int)
    y = np.array([[1,] if np.sum(itm)==1 else [0,] for itm in x], dtype=int)

    node_count = 4

    model = JIT_Network(input_shape=2, output_shape=1, node_count=node_count, learning_rate=0.1, id_num=10)
    model.add_connections(np.array([2,2,3,3,3]), np.array([0,1,0,1,2]))
    
    @njit
    def nopython_round(output_layer, expected):
        truth = 1
        for i, out in enumerate(output_layer):
            truth += not round(out) == expected[i]
        return truth == 1

    print("Training error: ", model.train(x, y, 10, 1000, 0.001))
    print("Validation accuracy: ", model.validate(x, y, nopython_round))
    print(model.predict(((1, 1), (1, 0), (0, 1), (0, 0))))
    # utils.display_network(model)

if __name__ == '__main__':
    run_xor_test()
