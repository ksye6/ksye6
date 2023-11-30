import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import yfinance as yf


np.random.seed(666)

#1
start_date = '2009-01-01'
end_date = '2018-12-31'

data = yf.download('^NYA', start=start_date, end=end_date)

#2
N=101
beta=0.5
s=2
m=2

price0 = data.loc[:,"Adj Close"].values # 长2515
price = price0[:-1] + beta * (price0[1:] - price0[:-1]) # 长2514

#3 初始化市场状态 长2515
mu=np.zeros(len(price0),int)

#4 决定市场状态 长2515
for i in range(1,len(price0)):
    b_t = 1 if price0[i] > price0[i-1] else 0
    mu[i] = (2*mu[i-1]+b_t) % (2**m)


#5 所有人策略可能性
all_strategies=np.random.randint(-1,2,size=(N,s,2**m))

#6 所有人每个策略的虚拟财富
virtual_wealth=np.zeros((N,s),float)+price0[0]*5

#7 所有人的财富
real_wealth=np.zeros(N,float)+price0[0]*5

#9 初始化市场
mu[0]=0

#10 每个人每次的策略选择，长N
real_choice=np.zeros(N,int)

#11 每个人的所有策略选择，长N
virtual_choice=np.zeros((N,s),int)

#12-14

# 所有人的每次模拟财富
recordings=[]
recordings.append(real_wealth.copy())

for i in range(len(price)-1): # 模拟2513次 0-2512
    
    # 初始化每次市场状态 和 所有人策略选择
    state=mu[i]
    choices=[] # -1/0/1
    
    # 获得每次的策略选择
    for agent in range(N):
        
        strategy=all_strategies[agent,:,state]
        
        if virtual_wealth[agent][0] > virtual_wealth[agent][1]:
            choices.append(strategy[0])
        
        elif virtual_wealth[agent][0] < virtual_wealth[agent][1]:
            choices.append(strategy[1])
        
        else:
            random_choose = np.random.randint(0,2)
            choices.append(strategy[random_choose])
    
    
    #8
    # 更新第一次每个人的策略选择 和 每个人的所有策略选择
    if i == 0:
        real_choice += np.array(choices,int)
        virtual_choice += all_strategies[:,:,state]
    
    
    # 更新每个人的财富 和 每个人的所有策略的虚拟财富
    real_wealth += real_choice*(price[i+1]-price[i])
    recordings.append(real_wealth.copy())
    
    virtual_wealth += virtual_choice*(price[i]-price[i-1])
    
    # 更新选择的策略 和 每个人所有策略的选择
    for agent in range(N):
        real_choice[agent] = np.max([np.min([real_choice[agent]+choices[agent], real_wealth[agent]/price[i+1]]),-real_wealth[agent]/price[i+1]])
        
        for j in range(s):
            virtual_choice[agent][j] = np.max([np.min([virtual_choice[agent][j]+all_strategies[agent,j,state],virtual_wealth[agent][j]/price[i+1]]),-virtual_wealth[agent][j]/price[i+1]])
    

totaldf=pd.DataFrame(np.array(recordings))


#15

plt.figure(figsize=(16,12))

plt.plot(totaldf.loc[:,np.argsort(real_wealth)[49]],linewidth=1.5,label='medium 49')
plt.plot(totaldf.loc[:,np.argsort(real_wealth)[50]],linewidth=1.5,label='median 50')
plt.plot(totaldf.loc[:,np.argsort(real_wealth)[51]],linewidth=1.5,label='median 51')

plt.plot(totaldf.loc[:,np.argsort(real_wealth)[-1]],linewidth=1.5,label='1st best')
plt.plot(totaldf.loc[:,np.argsort(real_wealth)[-2]],linewidth=1.5,label='2nd best')
plt.plot(totaldf.loc[:,np.argsort(real_wealth)[-3]],linewidth=1.5,label='3rd best')

plt.plot(totaldf.loc[:,np.argsort(real_wealth)[0]],linewidth=1.5,label='1st worst')
plt.plot(totaldf.loc[:,np.argsort(real_wealth)[1]],linewidth=1.5,label='2nd worst')
plt.plot(totaldf.loc[:,np.argsort(real_wealth)[2]],linewidth=1.5,label='3rd worst')

plt.plot(price0[:-1]*5, linewidth=1, color='black', label='market')
plt.legend()

plt.xlabel('time')
plt.ylabel('total wealth');

plt.show()


#16
sorted_wealth=real_wealth[np.argsort(real_wealth)]
wealth=[sorted_wealth[i] for i in [-1,-2,-3,49,50,51,0,1,2]]
gain=[sorted_wealth[i]/price[0]/5 for i in [-1,-2,-3,49,50,51,0,1,2]]
stock=[price0[-1]*5] * 9
compare=pd.DataFrame({'wealth':wealth,'gain':gain,'stock':stock},index=['best1','best2','best3','median1','median2','median3','worst1','worst2','worst3'])
compare


#17
# Advantage:
# 
# Potentially high returns: Based on the results, we can see that in some cases the "Wealth Game" strategy can achieve 
# relatively high returns (like the best three strategies).
# 
# Strategy diversity: In the data provided, we see that different "Wealth Game" strategies (best1, best2, best3) have achieved 
# relatively good performance. This suggests that this strategy may have some adaptability in different situations.

# Disadvantage:
# 
# High Risk: While some strategies may show higher returns, this also comes with higher risk. For example, the best three 
# strategies have relatively high returns, but they are also accompanied by higher volatility and greater risk.

# We can observe that the "Wealth Game" may perform differently in different time periods. Certain strategies may perform well 
# during certain periods and perform poorly during other periods. This may be affected by market conditions, stock performance 
# and strategy suitability.

# Whether we prefer to trade frequently in pursuit of high returns, or to buy stocks and hold them for the long term depends on 
# our personal investment goals, risk tolerance, and investment strategy.








