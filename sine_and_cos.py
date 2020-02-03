import numpy as np
import utils
import datetime
import evolutionary_module

np.random.seed(99)

with utils.OutSplit('sine-50k_{d.year}.{d.month}.{d.day}_{d.hour}.{d.minute}.{d.second}'.format(d=datetime.datetime.now())):
    x = (np.random.random(50000)*2*np.math.pi).reshape(50000, 1)
    y = (np.sin(x)+1).reshape(50000, 1)
    val_x = (np.random.random(1000)*2*np.math.pi).reshape(1000, 1)
    val_y = (np.sin(val_x)+1).reshape(1000, 1)

    network = evolutionary_module.evolve_node_count(x, y, val_x, val_y, utils.jit_near_compare, 20, 15, 600, 50, 0.95, 12)
    print(network.validate(val_x, val_y, utils.jit_near_compare))
    print(network.node_count, len(network.connections))

with utils.OutSplit('cos-50k_{d.year}.{d.month}.{d.day}_{d.hour}.{d.minute}.{d.second}'.format(d= datetime.datetime.now())):
    x = (np.random.random(50000)*2*np.math.pi).reshape(50000, 1)
    y = (np.cos(x)+1).reshape(50000, 1)
    val_x = (np.random.random(1000)*2*np.math.pi).reshape(1000, 1)
    val_y = (np.cos(val_x)+1).reshape(1000, 1)

    network = evolutionary_module.evolve_node_count(x, y, val_x, val_y, utils.jit_near_compare, 20, 15, 600, 50, 0.95, 12)
    print(network.validate(val_x, val_y, utils.jit_near_compare))
    print(network.node_count, len(network.connections))