import neuralnetwork.network as net
import random

model = net.Network(input_shape=2, output_shape=1, node_count=103, learning_rate=0.01, activation_func=net.leaky_relu)
model.add_connections(range(2, 102), range(0,2))
model.add_connections((-1,), range(2, 102))

x = [(1, 0) if random.random() < 0.25 else (0, 1) if random.random() < 0.5 else (1, 1) if random.random() < 0.75 else (0,0) for num in range(1000)]
y = [(1,) if sum(itm)==1 else (0,) for itm in x]

model.train(x, y, 50)

print([round(output[0]) for output in model.predict(((1, 1), (1, 0), (0, 1), (0, 0)))])