import homebrew.network as net
from ast import literal_eval
import numpy as np
import sys

def run_import():
    network = None
    with open('network.temp') as file:
        words = file.readline().split()
        if words[0] == 'Node' and words[1] == 'Count:':
            node_count = int(words[2])
        input_shape = int(file.readline())
        output_shape = int(file.readline())

        words = file.readline().split()
        if words[0] == 'Accuracy:':
            print(words[1:])
        
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