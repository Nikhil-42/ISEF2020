import numba
import numpy as np
from numba import njit
from numba.typed import List
import homebrew.network as net


@njit(parallel=True)
def spawn_populations(population_size, population_count, input_shape, output_shape, node_cap):   
    np.random.seed(42)
    
    populations = List()
    for i in range(population_count):
        node_count = np.random.randint(0, node_cap)
        population = List()
        for j in range(population_size):
            population.append(net.JIT_Network(input_shape, output_shape, node_count))
            node_count = population[0].node_count
            np.random.seed(len(population)+len(populations))
            connections = np.random.randint(0, high=node_count, size=(2, (node_count**2)/2)).astype(numba.int32)
            population[-1].add_connections(connections[0], connections[1])
        populations.append(population)
    return populations

@njit(parallel=True)
def evaluate_populations(x, y, val_x, val_y, post_processing, populations):
    np.random.seed(42)

    fitness = np.empty(shape=(len(populations), len(populations[0])), dtype=numba.float64)
    for i, population in enumerate(populations):
        for j, network in enumerate(population):
            traits = network.train(x, y, 1, 0.1, 1000, 0.001)
            if np.isnan(traits):
                fitness[i, j] = 0
            else:
                val = network.validate(val_x, val_y, post_processing)
                print(val)
                fitness[i, j] = val/(traits*len(network.connections))
    return fitness

if __name__ == "__main__":
    x = np.array([[1, 0] if rand < 0.25 else [0, 1] if rand < 0.5 else [1, 1] if rand < 0.75 else [0,0] for rand in np.random.random(60000)], dtype=int)
    y = np.array([[1,] if np.sum(itm)==1 else [0,] for itm in x], dtype=int)
    
    @njit
    def nopython_round(x):
        return round(x)

    populations = spawn_populations(10, 10, x.shape[1], y.shape[1], 150)
    fiteness = evaluate_populations(x, y, x, y, nopython_round, populations)
    print(fiteness)