import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#1
def update(x):
  x_n=(x[0]+x[-1])/2
  x[:-1]=(x[:-1]+x[1:])/2
  x[-1]=x_n
  return x

np.random.seed(1)
n=30
x=np.random.uniform(0,100,size=n)
y=np.random.uniform(0,100,size=n)

fig,ax=plt.subplots(figsize=(8,6), dpi=300)
ln,= plt.plot([], [], '-o', mfc='blue', color='black', linewidth=1, markersize=5,animated=True)
# ax.set_axis_off()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Polygon Plot')
ax.axis('off')

def init():
    ln.set_data([], [])
    return ln,

def update1(t):
    global x, y
    x=update(x)
    y=update(y)
    newx=np.append(x,x[0])
    newy=np.append(y,y[0])
    ln.set_data(newx,newy)
    ax.set_xlim([min(x)-(max(x)-min(x))/10, max(x)+(max(x)-min(x))/10])
    ax.set_ylim([min(y)-(max(y)-min(y))/10, max(y)+(max(y)-min(y))/10])
    return ln,

ani=FuncAnimation(fig=fig, #which figure you want to update
                  init_func=init,  #the inition function
                  func=update1, #the update function
                  frames=500,  #Source of data to pass func and each frame of the animation
                  interval=1, #delay between frames in milliseconds, 1000 means 1 second
                  blit=True #only update the plot that changed
                  )
# plt.show()
ani.save('C:\\Users\\张铭韬\\Desktop\\result_2.mp4', dpi=144, fps=24, extra_args=['-vcodec', 'libx264'])


#2
from scipy.interpolate import RectBivariateSpline
# 随机生成二维数组表示地形高度
np.random.seed(1)
N=30
X,Y=np.meshgrid(np.linspace(0, 10, N), np.linspace(0, 10, N))
Z=np.random.uniform(-0.7, 2.5, size=(N, N))

# 使用 RectBivariateSpline 函数平滑地形高度数据
f=RectBivariateSpline(np.linspace(0,10,N), np.linspace(0,10,N), Z, kx=3, ky=3)

smooth_N=100
smooth_X,smooth_Y=np.meshgrid(np.linspace(0,10, smooth_N), np.linspace(0,10, smooth_N))

smooth_Z=f(smooth_Y[:,0],smooth_X[0,:])

# 绘制等高线图和等高线填充图
L=[-1.5,-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3,3.5]
fig, ax=plt.subplots(figsize=(12,12), dpi=300)
cset1=ax.contour(smooth_X, smooth_Y, smooth_Z, colors='black', linewidths=0.3, corner_mask=False, levels=L)
cset2=ax.contourf(smooth_X, smooth_Y, smooth_Z, cmap='terrain',levels=L)


plt.clabel(cset1, levels=L,inline=True, fontsize=10, colors='black')

# 隐藏坐标轴
ax.set_xlim([0,5])
ax.set_ylim([0,5])
ax.set_axis_off()


plt.show()


#3
# 创建3D数组作为体素数据
Wall=np.zeros((14, 14, 14))
PillarDoor=np.zeros((14, 14, 14))
RoofStairs=np.zeros((14, 14, 14))
Window=np.zeros((14, 14, 14))

# 填充
Wall[3,5:11,0:8]=1
Wall[5:11,3,0:8]=1
Wall[12,5:11,0:8]=1
Wall[5:11,12,0:8]=1

Wall[3,7:9,2:6]=0
Wall[7:9,3,4:6]=0

PillarDoor[3,3:5,0:8]=1
PillarDoor[3,11:13,0:8]=1
PillarDoor[3,7:9,2:6]=1

PillarDoor[12,3:5,0:8]=1
PillarDoor[12,11:13,0:8]=1

PillarDoor[3:5,3,0:8]=1
PillarDoor[11:13,3,0:8]=1

PillarDoor[3:5,12,0:8]=1
PillarDoor[11:13,12,0:8]=1

RoofStairs[1:3,7:9,0]=1
RoofStairs[2,7:9,1]=1

RoofStairs[0:14,0:14,8]=1
RoofStairs[1:13,1:13,9]=1
RoofStairs[2:12,2:12,10]=1
RoofStairs[3:11,3:11,11]=1
RoofStairs[4:10,4:10,12]=1
RoofStairs[5:9,5:9,13]=1

Window[7:9,3,4:6]=1


fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

# 绘制
ax.voxels(Wall, facecolors='#DDDDDD')
ax.voxels(PillarDoor, facecolors='#9e6a08')
ax.voxels(RoofStairs, facecolors='#f2e0bd')
ax.voxels(Window, facecolors='#FFFFFF', edgecolor='#b6faf8', alpha=0.5, linewidth=1)
ax.view_init(10,-150)

ax.set_xlim(0,15)
ax.set_ylim(0,15)
ax.set_zlim(0,16)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_box_aspect([1,1,1])

plt.show()


#4
from skimage import measure

def taubin_heart(x,y,z):
    return (x**2+9*y**2/4+z**2-1)**3-x**2*z**3-9*y**2*z**3/80


# 网格
x=np.linspace(-2,2,100)
y=np.linspace(-2,2,100)
z=np.linspace(-2,2,100)
X,Y,Z=np.meshgrid(x,y,z)

# 计算函数值
F=taubin_heart(X,Y,Z)

# 提取网格表面
verts,faces,_,_=measure.marching_cubes(F,level=0)

# 顶点最大值和最小值
verts_min=np.min(verts,axis=0)
verts_max=np.max(verts,axis=0)

# 范围
x_range=verts_max[0]-verts_min[0]
y_range=verts_max[1]-verts_min[1]
z_range=verts_max[2]-verts_min[2]

# 缩放比例
scale_factor=2/max(x_range,y_range,z_range)

# 坐标
scaled_verts=(verts-verts_min)*scale_factor

# 中心点
centroid=np.mean(scaled_verts,axis=0)

# 平移顶点坐标以使中心点位于原点
translated_verts=scaled_verts-centroid

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

ax.plot_trisurf(translated_verts[:,1],translated_verts[:,0],faces,translated_verts[:,2],cmap='Spectral',alpha=0.8)

ax.view_init(15,-70)
ax.set_box_aspect([1,1,1])

ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])

plt.show()


#5
np.random.seed(1)

theta=np.linspace(0, 2*np.pi, 100)
a=np.cos(theta)
b=np.sin(theta)

fig,ax=plt.subplots(figsize=(12,12), dpi=300)
ln01, = plt.plot(a, b, '-', color='blue', linewidth=2,animated=True)
ln02, = plt.plot([0,0], [-1,1], '-', color='blue', linewidth=2, animated=True)
ln03, = plt.plot([-1,1], [0,0], '-', color='blue', linewidth=2, animated=True)

theta1 = np.linspace(2*np.pi,0,100)

lnl1, = plt.plot([], [], '-', color='red', linewidth=2,animated=True)
lnl2, = plt.plot([], [], '-', color='green', linewidth=2,animated=True)
lnl3, = plt.plot([], [], '-', color='blue', linewidth=2,animated=True)

lno1, = plt.plot([], [], 'o', mfc='red',markeredgecolor='none', markeredgewidth=0, markersize=5,animated=True)
lno2, = plt.plot([], [], 'o', mfc='green', markeredgecolor='none', markeredgewidth=0, markersize=5,animated=True)
lno3, = plt.plot([], [], '*', mfc='yellow', markeredgecolor='none', markeredgewidth=0,markersize=15,animated=True)

text_step=plt.text(2, 2, 't',color='black',fontsize=30)

# ax.set_axis_off()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Polygon Plot')
# ax.axis('off')
ax.set_xlim(-1, 3.2)
ax.set_ylim(-1, 3.2)

def init():
    return ln01,ln02,ln03,lnl1,lnl1,lnl1,lno1,lno2,text_step

i=0
def update(t):
    global i
    i=i+1
    text_step.set_text('t='+str(i))
    
    lnl1.set_data([0, np.cos(theta1[i])],[np.sin(theta1[i]),np.sin(theta1[i])])
    lnl2.set_data([np.cos(theta1[i]), np.cos(theta1[i])],[0,np.sin(theta1[i])])
    lnl3.set_data([0, np.cos(theta1[i])],[0,np.sin(theta1[i])])
    
    lno3.set_data(np.cos(theta1[i]),np.sin(theta1[i]))
    
    n=600
    x=np.linspace(np.cos(theta1[i]), np.cos(theta1[i])+5, n)
    y=np.linspace(np.sin(theta1[i]), np.sin(theta1[i])+5, n)
    
    lno1.set_data(x,np.sin(4*x+theta1[i]-4*np.cos(theta1[i])))
    lno2.set_data(np.sin(4*y+theta1[i]+np.pi/2-4*np.sin(theta1[i])),y)

    return ln01,ln02,ln03,lnl1,lnl1,lnl1,lno1,lno2,text_step

ani=FuncAnimation(fig=fig, #which figure you want to update
                  init_func=init,  #the inition function
                  func=update, #the update function
                  frames=99,  #Source of data to pass func and each frame of the animation
                  interval=1, #delay between frames in milliseconds, 1000 means 1 second
                  blit=True #only update the plot that changed
                  )
# plt.show()
ani.save('C:\\Users\\张铭韬\\Desktop\\result_5.mp4', dpi=144, fps=24, extra_args=['-vcodec', 'libx264'])



