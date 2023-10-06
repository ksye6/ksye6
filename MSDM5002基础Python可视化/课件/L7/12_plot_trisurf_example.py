from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
def f(x, y):
    return np.sqrt(x ** 2 + y ** 2)

radius=1;
num_points=1000
theta = 2 * np.pi * np.random.random(num_points)
r = np.random.random(num_points)

X = np.ravel(r * np.sin(theta))*radius
Y = np.ravel(r * np.cos(theta))*radius
Z = f(X, Y)*radius

ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='k');
#ax.scatter3D(X, Y, Z)
