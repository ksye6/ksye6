import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm


vertices = np.array([
    [-4.5, -4.5, -4.5],  # 顶点A
    [-4.5, 4.5, -4.5],   # 顶点B
    [4.5, 4.5, -4.5],    # 顶点C
    [4.5, -4.5, -4.5],   # 顶点D
    [-4.5, -4.5, 4.5],   # 顶点E
    [-4.5, 4.5, 4.5],    # 顶点F
    [4.5, 4.5, 4.5],     # 顶点G
    [4.5, -4.5, 4.5]     # 顶点H
])

edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],  # 底面边
    [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面边
    [0, 4], [1, 5], [2, 6], [3, 7]   # 连接底面和顶面的边
])


fig = plt.figure(figsize=(10,10), dpi=300)
ax = plt.axes(projection='3d',proj_type='ortho')

for edge in edges:
    x = vertices[edge, 0]
    y = vertices[edge, 1]
    z = vertices[edge, 2]
    ax.plot3D(x, y, z, 'b')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)
x = 4 * np.outer(np.cos(u), np.sin(v))
y = 4 * np.outer(np.sin(u), np.sin(v))
z = 4 * np.outer(np.ones(np.size(u)), np.cos(v)) + 9

# 绘制球体
ax.plot_surface(x, y, z, color='r', alpha=1)

# 设置坐标轴范围和刻度
ax.set_xlabel('X',fontsize=10,labelpad=8)
ax.set_ylabel('Y',fontsize=10,labelpad=8)
ax.set_zlabel('Z',fontsize=10,labelpad=8)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 13])
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.zaxis.set_tick_params(labelsize=10)

ax.set_title('A Cube')

# 设置视角
plt.axis('square')

theta=22
alpha=32
ax.view_init(theta, alpha)
ax.set_proj_type('ortho')
plt.draw()

plt.show()


####################################################
epsilon=10E-10
Z=1
N=3

def func_dxy(x,y,z):    
    r=np.sqrt(x**2+y**2+z**2)
    if r<epsilon:
        return 0
    rho=2*Z*r/N
    
    R3d=1/9/np.sqrt(30)*rho**2*Z**3/2*np.exp(-rho/2)
    Y3dxy=np.sqrt(60/4)*x*y/r**2*np.sqrt(1/4/np.pi)
    return Y3dxy*R3d
    
def func_px(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    if r<epsilon:
        return 0

    rho=2*Z*r/N
    R2p=(1/2/np.sqrt(6))*rho*Z**3/2*np.exp(-rho/2)
    Y2px=np.sqrt(3)*x/r/np.sqrt(4*np.pi)

    return R2p*Y2px

    
    

x_min=-20; x_max=20; z=0
y_min=-20; y_max=20

num_x=50; num_y=50
dxy=np.zeros([num_x,num_y],float); 
xx=np.zeros([num_x,num_y],float); yy=np.zeros([num_x,num_y],float)
px=np.zeros([num_x,num_y],float); 
colors=np.zeros([num_x,num_y],dtype=str); 
colortuple=('r','b','y','g')
for nx in range(num_x):
    for ny in range(num_y):
        xx[nx,ny]=nx/(num_x-1)*(x_max-x_min)+x_min
        yy[nx,ny]=ny/(num_y-1)*(y_max-y_min)+y_min
        dxy[nx,ny]=func_dxy(nx/(num_x-1)*(x_max-x_min)+x_min,ny/(num_y-1)*(y_max-y_min)+y_min,z)
        px[nx,ny]=func_px(nx/(num_x-1)*(x_max-x_min)+x_min,ny/(num_y-1)*(y_max-y_min)+y_min,z)
        colors[nx,ny] = colortuple[(nx + ny)%len(colortuple)]


ax=plt.axes(projection='3d',proj_type='ortho')
#ensure the aspect is correct
ax.set_box_aspect([1,1,1])

ax.plot_wireframe(xx, yy, px, color='grey')
#ax.plot_surface(xx, yy, px,  cmap='jet', linewidth=0,shade=False)
ax.contour3D(xx,yy,px,50,zdir='z',offset=0.045,cmap=cm.coolwarm)
ax.contour3D(xx,yy,px,50,zdir='x',offset=22,cmap=cm.coolwarm)
ax.contour3D(xx,yy,px,50,zdir='y',offset=22,cmap=cm.coolwarm)
ax.set_title('combination of different plots')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#plt.title('cstride=10')

phi=25
theta=80
ax.view_init(phi, theta)
plt.draw()

plt.show()
