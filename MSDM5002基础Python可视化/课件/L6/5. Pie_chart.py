import matplotlib.pyplot as plt
 
slices = [8,2,8,6]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']
 
plt.pie(slices, labels=activities,
  colors=cols,
  startangle=90,
  shadow= True,
  explode=(0,0,0.1,0),
  autopct='%1.2f%%')
 
plt.title('Pie Plot')
plt.show()