import matplotlib.pyplot as plt
# axes=plt.subplot(111)
# plt.plot([1,5],[2,3])
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# # axes.set_aspect(1)
# # plt.show()

plt.figure()
plt.subplot(111)
plt.plot([1,5],[2,3])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")



# axes=plt.gca() ###select the current axes
# axes.set_aspect(1)

# axes.set_aspect(7,adjustable='datalim')

# axes.set_aspect('equal')

# ### try others:"Scaled, Equal, tight, auto, image, square". 
# plt.axis('Image')  
# plt.show()