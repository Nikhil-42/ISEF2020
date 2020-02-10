import numpy as np
import evolutionary_module as evm
import utils

np.random.seed(57)

with utils.OutSplit('multiple_cancer') as out_split:
    cancer_data = np.genfromtxt('datasets\\breast-cancer-wisconsin-data.csv', delimiter=',', dtype=[('id', '<i4'), ('diagnosis', 'S1'), ('radius_mean', '<f8'), ('texture_mean', '<f8'), ('perimeter_mean', '<f8'), ('area_mean', '<f8'), ('smoothness_mean', '<f8'), ('compactness_mean', '<f8'), ('concavity_mean', '<f8'), ('concave_points_mean', '<f8'), ('symmetry_mean', '<f8'), ('fractal_dimension_mean', '<f8'), 
    ('radius_se', '<f8'), ('texture_se', '<f8'), ('perimeter_se', '<f8'), ('area_se', '<f8'), ('smoothness_se', '<f8'), ('compactness_se', '<f8'), ('concavity_se', '<f8'), ('concave_points_se', '<f8'), ('symmetry_se', '<f8'), ('fractal_dimension_se', '<f8'), ('radius_worst', '<f8'), ('texture_worst', '<f8'), ('perimeter_worst', '<f8'), ('area_worst', '<f8'), ('smoothness_worst', '<f8'), ('compactness_worst', '<f8'), ('concavity_worst', '<f8'), ('concave_points_worst', '<f8'), ('symmetry_worst', '<f8'), ('fractal_dimension_worst', '<f8')], names=True)
    x = np.column_stack((
        cancer_data['radius_mean'], cancer_data['texture_mean'], cancer_data['perimeter_mean'], cancer_data['area_mean'], cancer_data['smoothness_mean'], cancer_data['compactness_mean'], cancer_data['concavity_mean'], cancer_data['concave_points_mean'], cancer_data['symmetry_mean'], cancer_data['fractal_dimension_mean'],
        cancer_data['radius_se'], cancer_data['texture_se'], cancer_data['perimeter_se'], cancer_data['area_se'], cancer_data['smoothness_se'], cancer_data['compactness_se'], cancer_data['concavity_se'], cancer_data['concave_points_se'], cancer_data['symmetry_se'], cancer_data['fractal_dimension_se'],
        cancer_data['radius_worst'], cancer_data['texture_worst'], cancer_data['perimeter_worst'], cancer_data['area_worst'], cancer_data['smoothness_worst'], cancer_data['compactness_worst'], cancer_data['concavity_worst'], cancer_data['concave_points_worst'], cancer_data['symmetry_worst'], cancer_data['fractal_dimension_worst']
    ))
    x = x / np.max(x, axis=0)
    y = np.array([float(item==b'B') for item in cancer_data['diagnosis']]).reshape(len(x), 1)

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

    np.random.seed(r)

    perm = np.random.permutation(len(x))

    x = x[perm]
    y = y[perm]

    val_x = x[-100:]
    val_y = y[-100:]

    x = x[:-100]
    y = y[:-100]

    for i in range(50):
        best_cancer = evm.evolve_node_count(x, y, val_x, val_y, utils.jit_round_compare, population_count, population_size, node_cap, generations, target_accuracy, i*r+r)
