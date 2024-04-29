import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

### 1 Data Preprocessing 

# 港铁
S1 = yf.download("0066.HK", start="2008-01-01")
S1 = S1["Close"]

S1_past = S1[:int(len(S1)/4*3)+1]
S1_future = S1[int(len(S1)/4*3)+1:]

X1_past = np.log(np.array(S1_past[1:]) / np.array(S1_past[:-1]))

plt.figure(figsize=(12, 12))
plt.plot(S1_past)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price of 0066.HK")
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(X1_past)
plt.xlabel("Time")
plt.ylabel("Daily Returns")
plt.title("Daily Returns of 0066.HK")
plt.subplots_adjust(left=0.18)
plt.show()


# 载通
S2  = yf.download("0062.HK", start="2008-01-01")
S2 = S2["Close"]

S2_past = S2[:int(len(S2)/4*3)+1]
S2_future = S2[int(len(S2)/4*3)+1:]

X2_past = np.log(np.array(S2_past[1:]) / np.array(S2_past[:-1]))

plt.figure(figsize=(12, 12))
plt.plot(S2_past)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price of 0062.HK")
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(X2_past)
plt.xlabel("Time")
plt.ylabel("Daily Returns")
plt.title("Daily Returns of 0062.HK")
plt.subplots_adjust(left=0.18)
plt.show()


### 2 Mean-Variance Analysis

# 定义计算函数
def calculate_all(t, h):
    X1_returns = pd.Series(X1_past[max(t-h,0):t], name='港铁')
    X2_returns = pd.Series(X2_past[max(t-h,0):t], name='载通')
    X_returns = pd.concat([X1_returns,X2_returns],axis=1)
    mean_return = X_returns.mean()
    var_return = X_returns.var()
    cov_return =  X_returns.cov()
    p0 = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
    S0 = (p0 * S1_past[max(t-h,0):t]+(1-p0)*S2_past[max(t-h,0):t]).iloc[-1]
    # y1 = [np.mean(S0.pct_change().dropna())-0.02/360]/np.std(S0.pct_change().dropna())
    y0 = [p0 * np.mean(S1_past[max(t-h,0):t].pct_change().dropna())+(1-p0)*np.mean(S2_past[max(t-h,0):t].pct_change().dropna())-0.02/360] / (p0**2*var_return.iloc[0]+(1-p0)**2*var_return.iloc[1]+2*p0*(1-p0)*cov_return.iloc[0,1])**0.5
    return p0,S0,y0

# 定义绘制函数
def plot_all(h_values):
    for h in h_values:
        t_range = range(min(h_values), len(X1_past))
        p0_values = []
        S0_values = []
        y0_values = []
        for t in t_range:
            p0,S0,y0 = calculate_all(t, h)
            p0_values.append(p0)
            S0_values.append(S0)
            y0_values.append(y0)
        
        plt.figure(figsize=(12, 12))
        plt.plot(t_range, p0_values)
        plt.xlabel("t")
        plt.ylabel("P0(t, h)")
        plt.title(f"P0(t, h) for h = {h}")
        plt.show()
        
        plt.figure(figsize=(12, 12))
        plt.plot(t_range, S0_values)
        plt.xlabel("t")
        plt.ylabel("S0(t, h)")
        plt.title(f"S0(t, h) for h = {h}")
        plt.show()
        
        plt.figure(figsize=(12, 12))
        plt.plot(t_range, y0_values)
        plt.xlabel("t")
        plt.ylabel("y0(t, h)")
        plt.title(f"y0(t, h) for h = {h}")
        plt.show()

# 设置 h 值
h_values = [30, 100, 300, len(X1_past)]

# 绘制曲线
plot_all(h_values)

# The larger h is, the smoother all plots are.


### 3 Moving Average
import talib

## EMA
EMA30=talib.EMA(S1_past,timeperiod=30)
EMA100=talib.EMA(S1_past,timeperiod=100)
EMA300=talib.EMA(S1_past,timeperiod=300)

plt.figure(figsize=(12,12))
plt.plot(S1_past.index,EMA30,label='EMA30')
plt.plot(S1_past.index,EMA100,label='EMA100')
plt.plot(S1_past.index,EMA300,label='EMA300')
plt.title("EMA of 0066.HK Stock Price")
plt.legend()
plt.show() # The larger w is, the smoother EMA line is.

## SMA
SMA30=talib.SMA(S1_past,timeperiod=30)
SMA100=talib.SMA(S1_past,timeperiod=100)
SMA300=talib.SMA(S1_past,timeperiod=300)

plt.figure(figsize=(12,12))
plt.plot(S1_past.index,SMA30,label='SMA30')
plt.plot(S1_past.index,SMA100,label='SMA100')
plt.plot(S1_past.index,SMA300,label='SMA300')
plt.title("SMA of 0066.HK Stock Price")
plt.legend()
plt.show()

plt.figure(figsize=(12,12))
plt.plot(S1_past.index,SMA30,label='SMA30')
plt.plot(S1_past.index,EMA30,label='EMA30')
plt.title("Comparison between EMA and SMA (w = 30)")
plt.legend()
plt.show()

plt.figure(figsize=(12,12))
plt.plot(S1_past.index,SMA100,label='SMA100')
plt.plot(S1_past.index,EMA100,label='EMA100')
plt.title("Comparison between EMA and SMA (w = 100)")
plt.legend()
plt.show()

plt.figure(figsize=(12,12))
plt.plot(S1_past.index,SMA300,label='SMA300')
plt.plot(S1_past.index,EMA300,label='EMA300')
plt.title("Comparison between EMA and SMA (w = 300)")
plt.legend()
plt.show()

# quite similar

## MACD
diff, dea, macd = talib.MACD(S1_past[-300:], fastperiod=12, slowperiod=26, signalperiod=9)
pos_macd = np.where(macd > 0, macd, 0)
neg_macd = np.where(macd < 0, macd, 0)

fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

axs[0].plot(S1_past.index[-300:], S1_past[-300:], label='0066.HK')
axs[0].legend()

axs[1].plot(diff, label='diff', color='gray')
axs[1].plot(dea, label='dea', color='blue')
axs[1].bar(S1_past.index[-300:], pos_macd, width=0.6, color='red')
axs[1].bar(S1_past.index[-300:], neg_macd, width=0.6, color='green')
axs[1].axhline(y=0, color='black', linewidth=2)
axs[1].legend()

plt.show()


### 4 Probability Density Function
from scipy.optimize import minimize

# Normal distribution
X1_mu = np.mean(X1_past)
print(X1_mu)
X1_sigma_squared = np.var(X1_past)
print(X1_sigma_squared)

# Logistic distribution
def logistic_function(x, x_star, b):
    return b * np.exp(-b * (x - x_star)) / (1 + np.exp(-b * (x - x_star)))**2

def negative_log_likelihood(params, data):
    x_star, b = params
    ll = np.log(logistic_function(data, x_star, b)).sum()
    return -ll

initial_guess = [0, 1]
result = minimize(negative_log_likelihood, initial_guess, args=(X1_past,))
print(result)
X1_star, X1_b = result.x

print(X1_star)
print(X1_b)

# fits
from scipy.stats import norm
from scipy.stats import logistic

xranges = np.linspace(-0.05, 0.05, 100)
normal_cdf_values = norm.cdf(xranges, X1_mu, scale=np.sqrt(X1_sigma_squared))
logistic_cdf_values = 1 / (1 + np.exp(-X1_b * (xranges - X1_star))) # 最优拟合
# logistic_cdf_values = logistic.cdf(xranges, loc=np.mean(X1_past), scale=np.std(X1_past)) 库自带的函数偏差较大

plt.figure(figsize=(12,12))
plt.plot(xranges, normal_cdf_values, label='G(x) - Normal CDF')
plt.plot(xranges, logistic_cdf_values, label='L(x) - Logistic CDF')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.title('Fitted Functions')
plt.show()

# The two assumptions look reasonable

## 4.1 Digitization and Conditional Probability

epsilon = 0.004

# X to Y
def digitize_X(X, epsilon):
    Y = []
    for x in X:
        if x < -epsilon:
            Y.append('D')
        elif x > epsilon:
            Y.append('U')
        else:
            Y.append('H')
    return np.array(Y)

Y_digi = digitize_X(X1_past, epsilon)

# 计算条件累积分布函数 CDF[X(t) = x | Y(t+1) = y]
def conditional_cdf(X, Y, x, y, assump = True):
    filtered_X = X[:-1][Y[1:] == y]  # 选择 Y(t+1) = y 对应的 X(t)
    if assump == 'Logistic':
        return logistic.cdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X)) # 库自带的Logistic
    if assump == 'Normal':
        return norm.cdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X)) # 库自带的Normal
    return np.sum(filtered_X <= x) / len(filtered_X) # CDF定义

# 生成 x 值的范围
xranges = np.linspace(-0.05, 0.05, 100)

## 绘制 conditional CDF
plt.figure(figsize=(12,12))
for y in ['D', 'U', 'H']:
    cdf_values = [conditional_cdf(X1_past, Y_digi, x, y) for x in xranges]
    plt.plot(xranges, cdf_values, label=f'CDF[X(t) | Y(t+1) = {y}]')

plt.xlabel('x')
plt.ylabel('CDF')
plt.legend()
plt.title(f'Conditional CDF of X given Y with no assumption')
plt.show()

plt.figure(figsize=(12,12))
for y in ['D', 'U', 'H']:
    cdf_values = [conditional_cdf(X1_past, Y_digi, x, y,'Normal') for x in xranges]
    plt.plot(xranges, cdf_values, label=f'CDF[X(t) | Y(t+1) = {y}]')

plt.xlabel('x')
plt.ylabel('CDF')
plt.legend()
plt.title(f'Conditional CDF of X given Y with Normal assumption')
plt.show()

plt.figure(figsize=(12,12))
for y in ['D', 'U', 'H']:
    cdf_values = [conditional_cdf(X1_past, Y_digi, x, y,'Logistic') for x in xranges]
    plt.plot(xranges, cdf_values, label=f'CDF[X(t) | Y(t+1) = {y}]')

plt.xlabel('x')
plt.ylabel('CDF')
plt.legend()
plt.title(f'Conditional CDF of X given Y with Logistic assumption')
plt.show()

## 绘制 conditional PDF,选用正态假设
def conditional_pdf(X, Y, x, y):
    filtered_X = X[:-1][Y[1:] == y]
    return norm.pdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X))
    # return logistic.pdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X))

plt.figure(figsize=(12,12))
for y in ['D', 'U', 'H']:
    pdf_values = [conditional_pdf(X1_past, Y_digi, x, y) for x in xranges]
    plt.plot(xranges, pdf_values, label=f'PDF[X(t) | Y(t+1) = {y}]')

plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.title('Conditional PDF of X given Y with Normal assumption')
plt.show()


### 5 Bayes Detector

# prior probabilities q(y)
prior_yD = len(Y_digi[Y_digi=='D'])/len(Y_digi)
prior_yH = len(Y_digi[Y_digi=='H'])/len(Y_digi)
prior_yU = len(Y_digi[Y_digi=='U'])/len(Y_digi)
priors = {'D':prior_yD,'H':prior_yH, 'U':prior_yU}

def compute_y_star(X, Y):
    x = X[-1]  # 观测值 X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors[y] * conditional_pdf(X1_past, Y_digi, x, y)  # 计算 q(y) * f_y(x)
        if prob > max_prob:
            max_prob = prob
            y_star = y
    return y_star

y_star = compute_y_star(X1_past, Y_digi)

# 计算关键x值
y_star_list = []
critical_values = []
prev_y_star = None
for i in range(len(X1_past)):
    x = X1_past[i]
    y_star = compute_y_star(X1_past[:i+1], Y_digi[:i+1])
    y_star_list.append(y_star)
    if prev_y_star is not None and y_star != prev_y_star:
        critical_values.append(x)
    prev_y_star = y_star

# 在图上标记关键值
sorted_data = sorted(zip(X1_past, y_star_list))
sorted_X, sorted_Y = zip(*sorted_data)

diff_index = [i for i in range(1, len(sorted_Y)) if sorted_Y[i] != sorted_Y[i-1]]
diff_value = [sorted_X[i] for i in diff_index]
diff_type = [sorted_Y[i] for i in diff_index]

plt.figure(figsize=(12,12))
plt.plot(sorted_X, sorted_Y,color='red',alpha=0.2,linestyle='--')
plt.scatter(sorted_X, sorted_Y,s=20)
plt.scatter(diff_value, diff_type, marker='*', color='red', s=100)
for x, y in zip(diff_value, diff_type):
    plt.annotate(str(round(x,5)), xy=(x, y), xytext=(x-0.022, y), color='black')
plt.xlabel('x value')
plt.ylabel('y type')
plt.title('Posterior relationship')
plt.show()


### 6 Association Rules

# 构建数据集
tent = []
for i in range(len(Y_digi) - 6 + 1):
    row = Y_digi[i:i+6]
    tent.append(row)
# 创建DataFrame
df = pd.DataFrame(tent, columns=[f'Y_{i}' for i in range(1, 7)])
print(df)

df_encoded = pd.get_dummies(df)
df = df_encoded.astype(int)

from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001)
# 筛选符合规则形式的关联规则
rules = rules[rules['antecedents'].apply(lambda x: len(x) == 5) &rules['consequents'].apply(lambda x: len(x) == 1)]
filtered_rules = rules[~rules['antecedents'].apply(lambda x: 'Y_6_D' in x or 'Y_6_H' in x or 'Y_6_U' in x)]

# 按支持度排序
rules_support_top10 = filtered_rules.sort_values(by='support', ascending=False).head(10)
rules_support_top10['antecedents'] = rules_support_top10['antecedents'].apply(lambda x: sorted(x, key=lambda y: int(y.split('_')[1])))
rules_support_top10.loc[:, ['antecedents', 'consequents', 'support']]

# 按置信度排序
rules_confidence_top10 = filtered_rules.sort_values(by='confidence', ascending=False).head(10)
rules_confidence_top10['antecedents'] = rules_confidence_top10['antecedents'].apply(lambda x: sorted(x, key=lambda y: int(y.split('_')[1])))
rules_confidence_top10.loc[:, ['antecedents', 'consequents', 'confidence']]

# geometric mean
filtered_rules.loc[:,'geometric_mean'] = np.sqrt(filtered_rules.loc[:,'support'] * filtered_rules.loc[:,'confidence'])
rules_geometric_mean_top10 = filtered_rules.sort_values(by='geometric_mean', ascending=False).head(10)
rules_geometric_mean_top10['antecedents'] = rules_geometric_mean_top10['antecedents'].apply(lambda x: sorted(x, key=lambda y: int(y.split('_')[1])))
rules_geometric_mean_top10.loc[:, ['antecedents', 'consequents', 'geometric_mean']]

# RPF
filtered_rules.loc[:,'RPF'] = filtered_rules.loc[:,'support'] * filtered_rules.loc[:,'confidence']**2
rules_RPF_top10 = filtered_rules.sort_values(by='RPF', ascending=False).head(10)
rules_RPF_top10['antecedents'] = rules_RPF_top10['antecedents'].apply(lambda x: sorted(x, key=lambda y: int(y.split('_')[1])))
rules_RPF_top10.loc[:, ['antecedents', 'consequents', 'RPF']]

# Conviction and Lift can also provide some information

# 对比
RPF_result = []
for row in range(10):
    antecedents = rules_RPF_top10.iloc[row, 0]
    hdu = ''
    for i in antecedents:
        hdu += i.split('_')[2]
    hdu += next(iter(rules_RPF_top10.iloc[row, 1])).split('_')[2]
    RPF_result.append(hdu)
print(RPF_result)

gm_result = []
for row in range(10):
    antecedents = rules_geometric_mean_top10.iloc[row, 0]
    hdu = ''
    for i in antecedents:
        hdu += i.split('_')[2]
    hdu += next(iter(rules_geometric_mean_top10.iloc[row, 1])).split('_')[2]
    gm_result.append(hdu)
print(gm_result)

set(RPF_result).intersection(set(gm_result)) # 8 rules


### 7 A Portfolio with One Stock and Money
# 近几年股价太烂肯定亏，我们只用前一半的数据，即500项
# 前一半的股价图
plt.figure(figsize=(12, 12))
plt.plot(S1_future[:500])
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price of 0066.HK")
plt.show()

# see excel
def action(today,y_star,macd_new,delta):
    if (today == 'U' and y_star == 'U' and macd_new > 0 and delta > 0) or (today == 'H' and y_star == 'U') or (today == 'D' and (y_star == 'U' or y_star == 'H')):
        act = 'Buy'
    elif (today == 'U' and y_star == 'U' and macd_new < 0 and delta < 0) or (today == 'U' and (y_star == 'H' or y_star == 'D')) or (today == 'H' and y_star == 'D') or (today == 'D' and y_star == 'D' and macd_new < 0):
        act = 'Sell'
    else:
        act = 'Hold'
    return act


def trade_strategy(g, r=0.001/100):
    M = [100000]  # 初始资金
    N = [0]  # 初始股票数量
    S = [S1_past.iloc[-1]]  # 初始股票价格
    
    portfolio_values = [100000]  # 投资组合价值列表
    government_values = [100000] # 国债价值变化列表
    trades = []  # 交易记录列表
    
    S_past = S1_past.copy()
    S_true = S1_future.copy()[:500]
    X_past = X1_past.copy()
    
    for t in range(0, len(S_true)):
        
        Y_past = digitize_X(X_past, epsilon)
        
        y_star = compute_y_star(X_past, Y_past)  # 预测下一天的分类
        today = digitize_X([X_past[-1]], epsilon) # 今天的分类
        _, _, macd = talib.MACD(S_past, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_new = macd.iloc[-1] # 今天的macd值
        delta = macd.iloc[-1]-macd.iloc[-2] # 今天对比昨天的差值
        
        act = action(today,y_star,macd_new,delta) # 决定买还是卖还是持有
        
        # 根据预测结果决定操作
        if act == 'Buy':
            shares_to_buy = round(g * M[-1] / S[-1])  # 购买的股票数量, 取整
            N.append(N[-1]+shares_to_buy)  # 更新N(t)
            M.append(M[-1]-shares_to_buy*S[-1]) # 更新M(t)
            trades.append(('Buy', shares_to_buy, S))  # 记录交易
        if act == 'Sell':
            shares_to_sell = round(g * N[-1])  # 卖出股票的数量, 取整
            N.append(N[-1]-shares_to_sell)  # 更新N(t)
            M.append(M[-1]+shares_to_sell*S[-1]) # 更新M(t)
            trades.append(('Sell', shares_to_sell, S))  # 记录交易
        if act == 'Hold':
            trades.append(('Hold', 0, S))  # 记录交易
        
        portfolio_values.append(M[-1]+N[-1]*S[-1])  # 更新投资组合价值
        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S.append(S_true.iloc[t])  # 更新实际股价S(t)
        
        S_past = pd.concat([S_past, S_true.iloc[[t]]]) # 更新历史股价序列
        X_past = np.append(X_past, np.log(S_past.iloc[-1] / S_past.iloc[-2])) # 更新历史股价日收益率序列

    return portfolio_values, government_values


g_aggressive = 0.5  # 激进的贪婪度参数
g_middle = 0.2
g_conservative = 0.05  # 保守的贪婪度参数

Vt_aggressive, V0 = trade_strategy(g_aggressive)
Vt_middle, _ = trade_strategy(g_middle)
Vt_conservative, _ = trade_strategy(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive, label= 'g_a = '+ str(g_aggressive))
plt.plot(days, Vt_middle, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V0, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time')
plt.legend()
plt.show()



















