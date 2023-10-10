
#(1)
import matplotlib.pyplot as plt
import numpy as np

#1
n=15000
x=np.linspace(-20,20,n)
y=np.where(x!=0,np.sin(x)/x,0.5)

plt.figure(figsize=(8,6), dpi=100)
plt.scatter(x,y,marker=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y=sin(x)/x (x!=0)')
plt.show()

#2
n=15000
x=np.linspace(-5,5,n)
n_=np.floor(x)-(1-np.floor(x)%2)
y=x-n_

plt.figure(figsize=(8,6), dpi=100)
plt.scatter(x,y,marker=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y=x-n')
plt.show()

#(2)
# np.random.rand(2,3) #[0,1)随机浮点数
# np.random.random(size=(2,3)) #[0,1)随机浮点数
# np.random.randint(20,30,size=(2,3)) #随机整数

#1
def update(x):
  x_n=(x[0]+x[-1])/2
  x[:-1]=(x[:-1]+x[1:])/2
  x[-1]=x_n
  return x

def plot_polygon(n,i,a=0,b=100):
  x=np.random.uniform(a,b,size=n)
  y=np.random.uniform(a,b,size=n)
  
  for j in range(i):
    update(x)
    update(y)
  
  return np.array(list(zip(x,y)))


#2
np.random.seed(1)
test=plot_polygon(50,3000)
x=test[:,0]
y=test[:,1]

plt.figure(figsize=(8,6), dpi=100)

plt.plot(x,y)
plt.plot(x[[0,-1]],y[[0,-1]])
plt.title('Line Plot')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

# The shape look like an ellipse in general.


#(3)
import sympy as sp

def build_expression(n):
  x=sp.symbols('x1:%d' % (n+1))
  expr=sp.prod([xi**3 for xi in x])
  return expr

def H(*args):
  k=len(args)
  expr=build_expression(len(args))
  x=sp.symbols(f'x1:{k+1}')
  known_values={x[i]: args[i] for i in range(k)}
  hessian=np.zeros((k,k))
  for i in range(k):
    for j in range(k):
      tent=sp.diff(expr, x[i], x[j])
      hessian[i,j]=tent.evalf(subs=known_values)
  
  return hessian

H(1,2)
H(2,3,5)
H(1,4,2,5,3)

#(4)
import random
random.seed(3)

def is_inside_circle(x,y,circle):
  cx,cy,radius=circle
  return (x-cx)**2+(y-cy)**2<=radius**2

circles=[]
for i in range(5):
  cx=random.uniform(0,10)
  cy=random.uniform(0,10)
  radius=3
  circles.append((cx,cy,radius))

def estimate_area(circles,grid_size=0.01):
  count=0
  for x in range(0,1000):
    for y in range(0,1000):
      point_x=x*grid_size
      point_y=y*grid_size
      for circle in circles:
        if is_inside_circle(point_x,point_y,circle):
          count+=1
          break
  return count*grid_size**2


estimate_area(circles,0.01)


#(5)
#1
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_name="D://轴//工具//Angela2.png"
img=mpimg.imread(file_name)

plt.figure(figsize=(8,6),dpi=300)
plt.imshow(img)
plt.title('Angela')
plt.axis('off')
plt.show()

#2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve

# filter_matrix=np.array([[1,4,6,4,1],
#                         [4,16,24,16,4],
#                         [6,24,36,24,6],
#                         [4,16,24,16,4],
#                         [1,4,6,4,1]])/256
# 
# blurred_img=np.zeros_like(img)
# for i in range(3):
#   blurred_img[:,:,i]=convolve(img[:,:,i],filter_matrix)

blurred_img=gaussian_filter(img,sigma=1.3)

plt.figure(figsize=(8,6),dpi=300)
plt.imshow(blurred_img)
plt.title('Angela_blur')
plt.axis('off')
plt.show()


#3
gray_img=0.2125*img[:,:,0]+0.7154*img[:,:,1]+0.0721*img[:,:,2]

plt.figure(figsize=(8,6),dpi=300)
plt.imshow(gray_img,cmap='gray')
plt.title('Angela_gray')
plt.axis('off')
plt.show()

#4
from scipy.signal import convolve2d
mx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
my=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
Ix=convolve2d(gray_img,mx,mode='same')
Iy=convolve2d(gray_img,my,mode='same')

I=np.sqrt(Ix**2+Iy**2)

I=(I-np.min(I))/(np.max(I)-np.min(I))


fig=plt.figure(figsize=(16,8), dpi=300)

ax1=plt.subplot(131)

ax1.imshow(Ix, cmap='gray')
ax1.set_title('Ix')
ax1.axis('off')

ax2=plt.subplot(132)

ax2.imshow(Iy, cmap='gray')
ax2.set_title('Iy')
ax2.axis('off')

ax3=plt.subplot(133)

ax3.imshow(I, cmap='gray')
ax3.set_title('I')
ax3.axis('off')

plt.show()


#(6)
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style


plt.figure(figsize=(8,8), dpi=300)

ax1=plt.subplot(2,2,1)

theta=np.linspace(0, 2*np.pi, 100)
rho=1-np.sin(theta)
x=rho*np.cos(theta)
y=rho*np.sin(theta)

ax1.plot(x, y,color='blue',linewidth=4)
plt.axis('square')
plt.title('fat heart',fontsize=20)




ax2=plt.subplot(2,2,2)
x1=np.linspace(0,2,100)
y1=np.sqrt(1-(abs(x1)-1)**2)
x2=np.linspace(-2,0,100)
y2=np.sqrt(1-(abs(x1)-1)**2)

y11=np.arccos(1-abs(x1))-math.pi
y22=np.arccos(1-abs(x2))-math.pi

ax2.plot(x1, y1,color='red',linewidth=4)
ax2.plot(x2, y2,color='red',linewidth=4)
ax2.plot(x1, y11,color='red',linewidth=4)
ax2.plot(x2, y22,color='red',linewidth=4)

plt.title('better heart',fontsize=20)



ax3=plt.subplot(2,2,3)
x1=np.linspace(0,2,100)
y1=np.sqrt(1-(abs(x1)-1)**2)
x2=np.linspace(-2,0,100)
y2=np.sqrt(1-(abs(x1)-1)**2)

y11=np.arccos(1-abs(x1))-math.pi
y22=np.arccos(1-abs(x2))-math.pi

ax3.plot(x1, y1,color='red',linewidth=4)
ax3.plot(x2, y2,color='red',linewidth=4)
ax3.plot(x1, y11,color='red',linewidth=4)
ax3.plot(x2, y22,color='red',linewidth=4)

plt.title('better heart',fontsize=20,x=0.5, y=0.5,color="b",size=14)
plt.axis("off")




ax4=plt.subplot(2,2,4, projection='polar')

theta=np.linspace(0, 2*np.pi, 100)
rho=1-np.sin(theta)

ax4.plot(theta, rho,color='blue',linewidth=4)
plt.title('fat heart',x=0.5, y=0.6,color="b",size=14)
plt.show()

