import utils
import numpy as np
from homebrew.network import JIT_Network
from matplotlib import pyplot as plt

network = JIT_Network(1, 1, 10, 0.01, id_num=0)

for i in range(1, 9):
    network.add_connection(i, 0)
    network.add_connection(9, i)
    network.nodes[i, 2] = -i

x = np.arange(0, 5, 0.01)
y  = np.sin(x)

network.train(x.reshape(-1,1), y.reshape(-1, 1), 10000)

plt.plot(x, y)
plt.plot(x, network.predict(x.reshape(len(x), 1)).reshape(len(y)))
plt.show()
