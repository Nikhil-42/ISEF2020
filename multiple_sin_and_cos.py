import numpy as np
import evolutionary_module as evm
import utils

np.random.seed(62)

with utils.OutSplit('multiple_sin') as out_split:
    x = (np.random.random(50000)*2*np.math.pi).reshape(50000, 1)
    y = (np.sin(x)+1).reshape(50000, 1)
    val_x = (np.random.random(1000)*2*np.math.pi).reshape(1000, 1)
    val_y = (np.sin(val_x)+1).reshape(1000, 1)

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
        best = evm.evolve_node_count(x, y, val_x, val_y, utils.jit_near_compare, population_count, population_size, node_cap, generations, target_accuracy, i*r+r)

with utils.OutSplit('nultiple_cos'.format(d= datetime.datetime.now())):
    
    print('population_count: ' + str(population_count))
    print('population_size: ' + str(population_size))
    print('node_cap: ' + str(node_cap))
    print('generations: ' + str(generations))
    print('target_accuracy: {}%'.format(target_accuracy*100))
    print('random_number: ' + str(r))
    
    x = (np.random.random(50000)*2*np.math.pi).reshape(50000, 1)
    y = (np.cos(x)+1).reshape(50000, 1)
    val_x = (np.random.random(1000)*2*np.math.pi).reshape(1000, 1)
    val_y = (np.cos(val_x)+1).reshape(1000, 1)

    for i in range(50):
        best = evm.evolve_node_count(x, y, val_x, val_y, utils.jit_near_compare, population_count, population_size, node_cap, generations, target_accuracy, i*r+r-1)