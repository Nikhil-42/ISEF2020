import matplotlib.pyplot as plt
import network as net
import networkx as nx
import numpy as np
import traceback
import datetime
import pathlib
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
        self.file = open("".join(['data/console-logs/', filename, '.data']), 'w')
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
    return np.all((output_layer-expected)**2 < 10**(-1))

def draw_network(network):
    G = nx.Graph()
    G.add_edges_from(network.connections)
    
    labels = {}
    for i in range(network.input_shape):
        labels[i] = "I #" + str(i)
    for n, i in enumerate(range(network.node_count - network.output_shape, network.node_count)):
        labels[i] = "O #" + str(n)

    H = nx.relabel_nodes(G, labels)
    nx.draw_networkx(H)

def display_network(network):
    draw_network(network)
    plt.show()

def save_network(network, folder='default'):
    draw_network(network)

    filepath = 'data/figures/' + folder + '/'
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    filename = filepath + '{}_node_{}_connection_{d.year}.{d.month}.{d.day}_{d.hour}.{d.minute}.{d.second}'.format(network.node_count, len(network.connections),d=datetime.datetime.now()) + '.png'
    
    plt.savefig(filename)

def data_to_csv(filename=None):
    if filename == None:
        filename= input("File Name: ")

    with open('data\\console-logs\\'+filename+'.data', 'r') as file:
        with open('data\\csv\\'+filename+'.csv', 'w') as output:

            # population_count = int(file.readline().split()[-1])
            # population_size = int(file.readline().split()[-1])
            # node_cap = int(file.readline().split()[-1])
            # generations = int(file.readline().split()[-1])

            if filename[:8] == 'multiple':
                output.write('node_count, connection_count, accuracy\n')
                i = 0
                for line in file:
                    line = line.strip()
                    i += 1
                    if line[:11] == 'Node Count:':
                        output.write(line.split()[-1] + ',')
                    elif line == 'Weighted Connections:':
                        i = 0
                    elif line[:13] == 'Best Accuracy':
                        output.write(str(i-1) + ',' + line.split()[-2] + '\n')
            else:
                output.write('node_count, generation, fitness, population\n')
                for line in file:
                    line = line.strip()
                    if line[:12] == 'Population: ':
                        population = line.split()[1]
                        node_count = line.split()[-1]
                    elif line[:13] == 'Generation # ':
                        entries = line.split()
                        output.write(','.join([node_count, entries[2], entries[-1], population]) + '\n')

def display_from_csv(filename=None, title=None, color=None):
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.ticker import MaxNLocator

    if filename == None:
        filename = input("File Name: ")
    if title == None:
        title = input("Graph Title: ")
    if color == None:
        color = input("Color: ")

    data = np.genfromtxt('data\\csv\\' + filename + '.csv', names=True, delimiter=',')

    if filename[:8] == 'multiple':
        x = data['node_count'] + (np.random.random(len(data)) - 0.5)/5
        y = data['connection_count'] + (np.random.random(len(data)) - 0.5)/5

        plt.title(title)
        plt.xlabel('Node Count')
        plt.ylabel('Connection Count')

        plt.scatter(x, y, color=color)
    else:
        generations = np.max(data['generation'])

        x = np.linspace(np.min(data['population']), np.max(data['population']), len(np.unique(data['population'])))
        y = np.linspace(np.min(data['generation']), np.max(data['generation']), len(np.unique(data['generation'])))

        X, Y  = np.meshgrid(x, y)
        Z = griddata((data['population'], data['generation']), data['fitness'], (X, Y), method='cubic')

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(title)
        plt.xlabel('Population')
        plt.ylabel('Generation')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

def import_network():
    from ast import literal_eval    

    network = None
    with open('network.temp') as file:
        words = file.readline().split()
        if words[0] == 'Node' and words[1] == 'Count:':
            node_count = int(words[2])
        input_shape = int(file.readline())
        output_shape = int(file.readline())
        
        network = net.JIT_Network(input_shape, output_shape, node_count, id_num=-1)

        line = file.readline().rstrip()
        if line == 'Weighted Connections:':
            for line in file:
                chunks = line.split(':')
                if chunks[0] == 'Biases':
                    network.nodes[:, 2] = np.array(literal_eval(chunks[1].strip()))
                    break
                connection = literal_eval(chunks[0])
                weight = float(chunks[1])
                network.add_connection(*connection)
                network.set_weight(connection, weight)

        network.remove_connection((0, 0))
    return network