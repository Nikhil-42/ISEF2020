import json
import numpy as np
import matplotlib.pyplot as plt

with open('datasets/Caspian/training data.json') as json_file:
	data_to_save = json.load(json_file)

#if you want other data
'''#data types
data_to_save["positions"] = []
data_to_save["forces"] = []
data_to_save["electron_energy"] = []
data_to_save["forces_diff"] = []
data_to_save["electron_energy_diff"] = []'''


data_to_save["positions"] = np.array(data_to_save["positions"])
data_to_save["forces"] = np.array(data_to_save["forces"])

norm_vector = np.array([1,1,1])
norm_vector = norm_vector / np.sqrt(np.dot(norm_vector,norm_vector))

dx = np.sqrt(np.sum(np.square(data_to_save["positions"][2][0]-data_to_save["positions"][2][1]))) - np.sqrt(np.sum(np.square(data_to_save["positions"][1][0]-data_to_save["positions"][1][1]))) #don't mess with this

distances = []
forces1 = []
forces2 = []
energy1 = [0]
energy2 = [0]
binding_energy = [0]
for time in range(len(data_to_save["positions"])-1,-1,-1): #integrates from right to left
	distances.append(np.sqrt(np.sum(np.square(data_to_save["positions"][time][0]-data_to_save["positions"][time][1]))))
	forces1.append(data_to_save["forces"][time][0].dot(norm_vector))
	forces2.append(data_to_save["forces"][time][1].dot(norm_vector))

	energy1.append(energy1[-1] + dx*forces1[-1])
	energy2.append(energy2[-1] + dx*forces2[-1])
	
	binding_energy.append(data_to_save["electron_energy"][time])

energy1 = energy1[1:]
energy2 = energy2[1:]

distances = np.array(distances) - dx/2
forces1 = np.array(forces1)
forces2 = np.array(forces2)
energy1 = np.array(energy1)
energy2 = np.array(energy2)

energy2 = energy2 * 27.211386245988 #convert from one non SI unit to another non SI unit

x = distances.reshape(-1, 1)
y = energy2.reshape(-1, 1) ##TODO if you want to scale from 0 - 1 like this, make sure to unscale at the end

import evolutionary_module as evm
from homebrew.network import JIT_Network
import utils

network = JIT_Network(1, 1, 152, 0.001, 0)

for i in range(50):
	network.add_connection(i+1, 0)

for i in range(50):
	for j in range (50):
		network.add_connection(j+51, i+1)
for i in range(50):
	for j in range (50):
		network.add_connection(j+101, i+51)

for i in range(50):
	network.add_connection(151, 101+i)

network.train(x, y, 10000, 100, 0.0001)

##do your NN stuff here
plt.plot(distances, energy2)
plt.plot(distances, network.predict(x).reshape(-1))
plt.show()
utils.display_network(network)