import numpy as np
import random
import matplotlib.pyplot as plt

#a
sample=5

def simulation(n=5000,iteration=15000):

    mid=int(n/2)
    wealth=np.ones(n)
    save=np.random.power(0.3,n)
    arr=np.arange(n)
    
    for i in range(iteration):
        
        np.random.seed(2*i)
        pair=np.random.permutation(arr)
        dice=np.random.rand(mid)
        pair1=pair[:mid]
        pair2=pair[mid:]
        
        gain_pair1=(dice-1)*(1-save[pair1])*wealth[pair1]+dice*(1-save[pair2])*wealth[pair2]
        gain_pair2=(dice-1)*(1-save[pair2])*wealth[pair2]+dice*(1-save[pair1])*wealth[pair1]
        
        wealth[pair1]+=gain_pair1
        wealth[pair2]+=gain_pair2
    
    return wealth

np.random.seed(123)
wdis=simulation()


#b
log_bin=[10.**x for x in np.linspace(np.log10(min(wdis)),np.log10(max(wdis)),101)]

fig, ax = plt.subplots()

res_log = ax.hist(wdis, density=1, cumulative=-1, histtype='step', log=True, bins=log_bin)
ax.set_title('Histogram of log(wealth)')
ax.set_xlabel('log(wealth)')
ax.set_ylabel('CCDF of wealth')
ax.set_xscale('log')

# plt.show()

##################### Another method ######################

sorted_wealth = np.sort(wdis)
ccdf = 1.0 - np.arange(len(wdis)) / len(wdis)

ax.loglog(sorted_wealth, ccdf)
# plt.show()

###########################################################

x = np.log(sorted_wealth[-400:])
y = np.log(ccdf[-400:])
coeffs = np.polyfit(x, y, 1)  # [a, b], y = ax + b
exponent = -coeffs[0]  # 幂律指数为斜率的相反数
print(exponent)
# 绘制拟合直线
x_fit = np.linspace(np.min(sorted_wealth), np.max(sorted_wealth), 101)
y_fit = np.exp(coeffs[1]) * np.power(x_fit, -exponent)
ax.loglog(x_fit, y_fit, color='red', linestyle='--')

plt.show()
# y = exp(-1.0905) * x^(-1.44044238)

#c
wlist = np.arange(2, 10, 0.05)
factorx = []

for w in wlist:

    index_w = np.abs(sorted_wealth - w).argmin()
    index_2w = np.abs(sorted_wealth - 2*w).argmin()
    
    ccdf_w = ccdf[index_w]
    ccdf_2w = ccdf[index_2w]
    
    x = (ccdf_w - ccdf_2w) / ccdf_w
    
    factorx.append(x)


fig = plt.figure()

factor = plt.plot(wlist, factorx)

plt.title('Reduce Factor x for wealthier people')
plt.xlabel('w')
plt.ylabel('x')

plt.show()

# From the fit formulation: y = exp(-1.0905) * x^(-1.44044238), x should be around 1-2**(-1.0905) = 0.53 (the reduced number is 0.53*n)


#d
wdis2=wdis[wdis<1]

lin_bin=np.linspace(0.,1.,101)

fig, ax = plt.subplots()

res_lin = ax.hist(wdis2, density=1, cumulative=-1, histtype='step', log=True, bins=lin_bin)
ax.set_title('Wealth distribution among poor agents')
ax.set_xlabel('wealth')
ax.set_ylabel('CCDF of wealth')

# plt.show()

##################### Another method ######################

sorted_wealth2 = np.sort(wdis2)
ccdf2 = 1.0 - np.arange(len(wdis2)) / len(wdis2)

ax.plot(sorted_wealth2, ccdf2)

# plt.show()

###########################################################

x2 = sorted_wealth2[:2000]
y2 = np.log(ccdf2[:2000])
coeffs2 = np.polyfit(x2, y2, 1)  # [a, b], y = ax + b
print(coeffs2)
# 绘制拟合直线
x_fit2 = np.linspace(np.min(sorted_wealth2), np.max(sorted_wealth2), 101)
y_fit2 = np.exp(coeffs2[0]*x_fit2 + coeffs2[1])
ax.plot(x_fit2, y_fit2, color='red', linestyle='--')

plt.show()

# y = exp(-1.78963301*x + 0.08882959)



#e

wlist2 = np.arange(0.02, 0.6, 0.02)
numbery = []

for w in wlist2:

    index_w = np.abs(sorted_wealth2 - w).argmin()
    ccdf_w = ccdf2[index_w]
    ccdf_2w = ccdf_w/2
    
    index_2w = np.abs(ccdf2 - ccdf_2w).argmin()
    
    y = (sorted_wealth2[index_2w] - w)
    
    numbery.append(y)


fig = plt.figure()

factor = plt.plot(wlist2, numbery,marker='o')

plt.title('Increase number y for less wealthy agents')
plt.xlabel('w')
plt.ylabel('y')

plt.show()

# From the fit formulation: y = exp(-1.78963301*x + 0.08882959), y should be = np.log(0.5)/-1.78963301 = 0.387


#f

cumulative_wealth=np.cumsum(sorted_wealth[::-1])
total_wealth= sum(sorted_wealth)

seg=int(len(wdis)*0.2)

percentage=cumulative_wealth[seg]/total_wealth

print(percentage)

# the percentage of wealth owned by the wealthiest 20% of the agents in my model is 0.75622



