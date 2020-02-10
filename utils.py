import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import numba
import sys

@numba.njit
def jit_to_categorical(ndarray, classes: int) -> np.ndarray:
    return np.array([[0 if val!=i else 1 for i in range(classes)] for val in ndarray])

@numba.njit
def jit_from_categorical(ndarray: np.ndarray) -> np.ndarray:
    outputs = np.empty(0)
    for arr in ndarray:
        outputs = np.append(outputs, (arr == max(arr)).nonzero())
    return outputs.astype(int)

class OutSplit(object):
    def __init__(self, filename, timestamp=True):
        if timestamp:
            filename += '_{d.year}.{d.month}.{d.day}_{d.hour}.{d.minute}.{d.second}'.format(d=datetime.datetime.now())
        self.file = open("".join(['data/', filename, '.data']), 'w')
        self.stdout = sys.stdout
        print("Console is logging at " + filename)

    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

@numba.njit
def jit_categorical_compare(output_layer, expected):
    return output_layer.argsort()[-1]==expected.argsort()[-1]

@numba.njit
def jit_round_compare(output_layer, expected):
    truth = 1
    for i, out in enumerate(output_layer):
        truth += not round(out) == expected[i]
    return truth == 1

@numba.jit
def jit_near_compare(output_layer, expected):
    return np.all((output_layer-expected)**2 < 10**(-3))

def display_network(network):
    G = nx.Graph()
    G.add_edges_from(network.connections)

    print(G.nodes())
    print(G.edges())

    labels = {}
    for i in range(network.input_shape):
        labels[i] = "I #" + str(i)
    for n, i in enumerate(range(network.node_count - network.output_shape, network.node_count)):
        labels[i] = "O #" + str(n)

    H = nx.relabel_nodes(G, labels)
    nx.draw_networkx(H)
    plt.savefig('{}_node_{}_connection_.png'.format(network.node_count, len(network.connections)))
    plt.show()