import numpy as np
import evolutionary_module as evm
import utils

np.random.seed(57)

with utils.OutSplit('multiple_xor') as out_split:
    x = np.array([[1, 0] if rand < 0.25 else [0, 1] if rand < 0.5 else [1, 1] if rand < 0.75 else [0,0] for rand in np.random.random(60000)], dtype=int)
    y = np.array([[1,] if np.sum(itm)==1 else [0,] for itm in x], dtype=int)

    # Grab input from the user and print it to stdout
    population_count = int(input('population_count: '))
    print('{}'.format(population_count))
    population_size = int(input('poplation_size: '))
    print('{}'.format(population_size))
    node_cap = int(input('node_cap: '))
    print('{}'.format(node_cap))
    generations = int(input('generations: '))
    print('{}'.format(generations))
    target_accuracy = float(input('target_accuracy: '))
    print('{}%'.format(target_accuracy*100))
    r = int(input("random_seed: "))
    print(str(r))

    for i in range(50):
        best_xor = evm.evolve_node_count(x, y, x, y, utils.jit_round_compare, population_count, population_size, node_cap, generations, target_accuracy, i*r+r)
        print("Best Network Accuracy:", best_xor.validate(x, y, utils.jit_round_compare)*100, "%")
