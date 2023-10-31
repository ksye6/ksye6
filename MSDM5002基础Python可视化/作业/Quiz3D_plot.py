from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10,10), dpi=100)
ax1=plt.subplot(111,projection='3d',proj_type='ortho')

# axs = fig.add_gridspec(2, 1, hspace=0, wspace=0)
# ax1=fig.add_subplot(axs[:,:],projection='3d',proj_type='ortho')


#plot the target
theta=np.linspace(0, np.pi*2, 500)
for nr in range(4):
    x = np.sin(theta)*nr*10
    y = np.cos(theta)*nr*10
    ax1.scatter3D(x,5,y,c='r',s=1)

#plot the spiral arrow
Num_period=10
Num_points=500
# Data for a three-dimensional line
zline = np.linspace(-np.pi*Num_period,0, Num_points)
xline = np.sin(zline)*zline
yline = np.cos(zline)*zline

ax1.plot3D(xline, zline, yline,'gray')
ax1.plot3D(xline*0, zline*3, yline*0, linewidth=2,color='b')
ax1.scatter3D(xline, zline, yline, c=abs(zline), s=abs(zline), cmap='jet');

phi=0
theta=-15
ax1.view_init(phi, theta)

ax1.text(0,-70,2,'spiral arrow',fontsize=20)
#ensure the aspect is correct
ax1.set_box_aspect([1,5,1])

plt.axis('off')

plt.draw()
plt.show()


