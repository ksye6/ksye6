import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4),dpi=200)
plt.subplot(111,projection='polar')
plt.axis('off')

r = np.arange(50,200,10)
theta = np.linspace(0,2*np.pi,len(r),endpoint=False)

angle_scaler=theta[1]*0.99
plt.bar(theta,r,
        angle_scaler,
        color=plt.cm.jet(np.linspace(0,1,len(r))),
        bottom=50)

plt.text(3.3,35,'Center',fontsize=14)

for ni in range(len(r)):
    plt.text(theta[ni],r[ni]+65,str(r[ni]),fontsize=10)