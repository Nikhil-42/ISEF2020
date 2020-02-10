from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

filename = input("Filename: ")
data = np.genfromtxt('data\\csv\\' + filename + '.csv', names=True, delimiter=',')

if filename[:8] == 'multiple':
    x = data['node_count'] + (np.random.random(len(data)) - 0.5)/5
    y = data['connection_count'] + (np.random.random(len(data)) - 0.5)/5

    plt.title(input("Graph Title: "))
    plt.xlabel('Node Count')
    plt.ylabel('Connection Count')

    color = input("Color: ")
    # plt.ylim(3, 8)
    # plt.xlim(3, 7)
    plt.scatter(x, y, color=color)
else:
    generations = np.max(data['generation'])

    x = np.linspace(np.min(data['population']), np.max(data['population']), len(np.unique(data['population'])))
    y = np.linspace(np.min(data['generation']), np.max(data['generation']), len(np.unique(data['generation'])))
    # print('x:', x)
    # print('y:', y)

    X, Y  = np.meshgrid(x, y)
    # print('X:', X)
    # print('Y', Y)
    Z = griddata((data['population'], data['generation']), data['fitness'], (X, Y), method='cubic')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(input("Graph Title: "))
    plt.xlabel('Population')
    plt.ylabel('Generation')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()