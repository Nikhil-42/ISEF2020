import evolutionary_module as evm
from homebrew.network import JIT_Network
import numpy as np
import utils

cancer_data = np.genfromtxt('datasets\\breast-cancer-wisconsin-data.csv', delimiter=',', dtype=[('id', '<i4'), ('diagnosis', 'S1'), ('radius_mean', '<f8'), ('texture_mean', '<f8'), ('perimeter_mean', '<f8'), ('area_mean', '<f8'), ('smoothness_mean', '<f8'), ('compactness_mean', '<f8'), ('concavity_mean', '<f8'), ('concave_points_mean', '<f8'), ('symmetry_mean', '<f8'), ('fractal_dimension_mean', '<f8'), 
('radius_se', '<f8'), ('texture_se', '<f8'), ('perimeter_se', '<f8'), ('area_se', '<f8'), ('smoothness_se', '<f8'), ('compactness_se', '<f8'), ('concavity_se', '<f8'), ('concave_points_se', '<f8'), ('symmetry_se', '<f8'), ('fractal_dimension_se', '<f8'), ('radius_worst', '<f8'), ('texture_worst', '<f8'), ('perimeter_worst', '<f8'), ('area_worst', '<f8'), ('smoothness_worst', '<f8'), ('compactness_worst', '<f8'), ('concavity_worst', '<f8'), ('concave_points_worst', '<f8'), ('symmetry_worst', '<f8'), ('fractal_dimension_worst', '<f8')], names=True)
x = np.column_stack((
    cancer_data['radius_mean'], cancer_data['texture_mean'], cancer_data['perimeter_mean'], cancer_data['area_mean'], cancer_data['smoothness_mean'], cancer_data['compactness_mean'], cancer_data['concavity_mean'], cancer_data['concave_points_mean'], cancer_data['symmetry_mean'], cancer_data['fractal_dimension_mean'],
    cancer_data['radius_se'], cancer_data['texture_se'], cancer_data['perimeter_se'], cancer_data['area_se'], cancer_data['smoothness_se'], cancer_data['compactness_se'], cancer_data['concavity_se'], cancer_data['concave_points_se'], cancer_data['symmetry_se'], cancer_data['fractal_dimension_se'],
    cancer_data['radius_worst'], cancer_data['texture_worst'], cancer_data['perimeter_worst'], cancer_data['area_worst'], cancer_data['smoothness_worst'], cancer_data['compactness_worst'], cancer_data['concavity_worst'], cancer_data['concave_points_worst'], cancer_data['symmetry_worst'], cancer_data['fractal_dimension_worst']
))
x = x / np.max(x, axis=0)
y = np.array([float(item==b'B') for item in cancer_data['diagnosis']]).reshape(len(x), 1)


with utils.OutSplit('breast-cancer'):
    population_count = 15
    population_size = 15
    node_cap = 100
    generations = 50
    target_accuracy = 0.97
    r = 0

    np.random.seed(r)

    perm = np.random.permutation(len(x))
    #np.random.shuffle(x)
    #np.random.shuffle(y)

    x = x[perm]
    y = y[perm]

    val_x = x[-100:]
    val_y = y[-100:]

    in_x = x[:-200]
    in_y = y[:-200]
    
    cross_val_x = x[-200:-100]
    cross_val_y = y[-200:-100]

    print('population_count: ' + str(population_count))
    print('population_size: ' + str(population_size))
    print('node_cap: ' + str(node_cap))
    print('generations: ' + str(generations))
    print('target_accuracy: {}%'.format(target_accuracy*100))
    print('random_seed: ' + str(r))

    best_network = evm.evolve_node_count(in_x, in_y, val_x, val_y, utils.jit_round_compare, population_count, population_size, node_cap, generations, target_accuracy, r)
    print(best_network.validate(cross_val_x, cross_val_y, utils.jit_round_compare))
    utils.display_network(best_network, "wisconsin-breast-cancer")