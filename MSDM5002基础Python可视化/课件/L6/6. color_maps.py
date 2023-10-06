import numpy as np
#import matplotlib.pylab as pl
import matplotlib.pyplot as plt

plt.figure()
x = np.linspace(0, 2*np.pi, 64)
y = np.cos(x) 

#plt.figure()
#plt.plot(x,y)

n = 11
# colors = plt.cm.seismic(np.linspace(0,1,n))
colors = plt.cm.bwr(np.linspace(0,1,n))

for i in range(n):
    colors[i,3]=1
    
# colors[1,3]=1
# colors[n-1,3]=1

for i in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
    plt.plot(x, i*y, color=colors[i+5])
    
plt.title('using colormap to plot many lines with different colors')