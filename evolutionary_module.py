import numba
import random
import numpy as np
from numba import njit, prange
from numba.typed import List
import homebrew.network as net

@njit
def spawn_populations(population_size, population_count, input_shape, output_shape, node_cap):   
    #np.random.seed(42)
    # random.seed(42)
    
    populations = List()

    for i in range(population_count):
        populations.append(spawn_population(population_size, input_shape, output_shape, node_cap))
    return populations

@njit
def spawn_population(population_size, input_shape, output_shape, node_cap):
    node_count = np.random.randint(0, node_cap)
    population = List()
    np.random.seed(node_count*30)
    
    for j in range(population_size):
        new_net = net.JIT_Network(input_shape, output_shape, node_count, np.random.random()/node_count)
        node_count = new_net.node_count
        np.random.seed(len(population)+node_count)
        for node in range(1, output_shape+1):
            to_node = node_count-node
            while input_shape <= to_node:
                from_node = to_node - np.random.randint(low=node_count/10, high=node_count/2+1)
                if from_node < 0:
                    from_node = np.random.randint(low=0, high=input_shape)
                new_net.add_connection(to_node, from_node)
                to_node = from_node
        connections = np.random.randint(0, high=node_count, size=(2, (node_count**2)/2)).astype(numba.int32)
        new_net.add_connections(connections[0], connections[1])
        population.append(new_net)
    return population

@njit
def evaluation(x, y, val_x, val_y, compare, population):
    fitnesses = np.empty(shape=len(population), dtype=numba.float64)
    for j, network in enumerate(population):
        traits = network.train(x, y, 1, 1000, 0.001)
        if np.isnan(traits):
            fitnesses[j] = -1
        else:
            val = network.validate(val_x, val_y, compare)
            fitnesses[j] = val/(traits*len(network.connections))*1000
    return fitnesses

@njit
def selection(num_pairs, fitnesses, population, r):
    # np.random.seed(int(r*100))

    population_size = len(population)

    rank_list = np.empty_like(fitnesses, dtype=np.int32)
    rank_sum = np.sum(np.arange(1, population_size+1))

    pairs = np.empty((num_pairs, 2), dtype=np.int32)
    pairs.fill(-1)
    parents = np.random.random((num_pairs, 2)) * rank_sum
    # print(parents, r)
    rank_list = fitnesses.argsort()[::-1]
    for p in range(num_pairs):
        for j in range(2):
            counter = 0
            prob = 0
            while parents[p, j] > prob and counter < len(rank_list):
                prob += population_size-counter
                if pairs[p, 0] == rank_list[counter]:
                    counter += 1
                counter += 1
            pairs[p, j] = rank_list[counter-1]

    return pairs

@njit
def recombination(input_shape, output_shape, pairs, population):
    random.randint(0, 1)
    new_networks = List()

    for i in range(len(pairs)):
        mom = population[pairs[i, 0]]
        dad = population[pairs[i, 1]]
        connections = mom.connections | dad.connections
        for j in range(len(connections)/2):
            connections.pop()
        new_net = net.JIT_Network(input_shape, output_shape, mom.node_count, 0.5*(mom.learning_rate+dad.learning_rate))
        new_net.set_connections(connections)
        for connection in connections:
            if connection in mom.connections:
                new_net.set_weight(connection, mom.get_weight(connection))
            else:
                new_net.set_weight(connection, dad.get_weight(connection))
        new_networks.append(new_net)
    return new_networks

@njit
def mutation(population, mutation_chances):
    for network in population:
        for rand in np.random.random(mutation_chances):
            if rand < 0.30:
                network.add_connection(np.random.randint(0, network.node_count), np.random.randint(0, network.node_count))
            if rand < np.random.random():
                network.learning_rate += (rand-0.5)/network.node_count

@njit
def competition(fitnesses, population, new_fitnesses, new_networks):
    for i, new_net in enumerate(new_networks):
        rank_list = fitnesses.argsort()
        if new_fitnesses[i] > fitnesses[rank_list[0]]:
            fitnesses[rank_list[0]] = new_fitnesses[i]
            population[rank_list[0]] = new_net


def evolution(x, y, compare, poplation_size, population_count, node_cap):
    random.seed(1)

    populations = spawn_populations(poplation_size, population_count, x.shape[1], y.shape[1], node_cap)

    for i in prange(population_count):
        population = populations[i]
        fitnesses = evaluation(x, y, x, y, compare, population)
        for i in range(50):
            pairs = selection(2, fitnesses, population, random.random())
            new_networks = recombination(x.shape[1], y.shape[1], pairs, population)
            mutation(new_networks, 10)
            new_fitnesses = evaluation(x, y, x, y, compare, new_networks)
            print(fitnesses)
            competition(fitnesses, population, new_fitnesses, new_networks)
        best = population[fitnesses.argsort()[-1]]
        subset = x[:10]
        print(subset, "\n", best.predict(subset))
        print(best.validate(x, y, compare))
        print(best.connections, best.learning_rate)

if __name__ == "__main__":
    x = np.array([[1, 0] if rand < 0.25 else [0, 1] if rand < 0.5 else [1, 1] if rand < 0.75 else [0,0] for rand in np.random.random(60000)], dtype=int)
    y = np.array([[1,] if np.sum(itm)==1 else [0,] for itm in x], dtype=int)
    
    @njit
    def nopython_round(output_layer, expected):
        truth = 1
        for i, out in enumerate(output_layer):
            truth += not round(out) == expected[i]
        return truth == 1

    evolution(x, y, nopython_round, 5, 5, 10)
