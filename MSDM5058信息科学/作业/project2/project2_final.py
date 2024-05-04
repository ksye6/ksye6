import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
# akshare as ak
# 获取基金价格ak.fund_open_fund_info_em(symbo]="380005",indicator="单位净值走势")
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

# 调整下预测函数
def compute_y_star(X, Y):
    x = X[-1]  # 观测值 X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors[y] * conditional_pdf(X, Y, x, y)  # 计算 q(y) * f_y(x)
        if prob > max_prob:
            max_prob = prob
            y_star = y
    return y_star


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
            trades.append(('Buy', shares_to_buy, S[-1]))  # 记录交易
        if act == 'Sell':
            shares_to_sell = round(g * N[-1])  # 卖出股票的数量, 取整
            N.append(N[-1]-shares_to_sell)  # 更新N(t)
            M.append(M[-1]+shares_to_sell*S[-1]) # 更新M(t)
            trades.append(('Sell', shares_to_sell, S[-1]))  # 记录交易
        if act == 'Hold':
            trades.append(('Hold', 0, S[-1]))  # 记录交易
        
        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S.append(S_true.iloc[t])  # 更新实际股价S(t)
        portfolio_values.append(M[-1]+N[-1]*S[-1])  # 更新投资组合价值
        
        S_past = pd.concat([S_past, S_true.iloc[[t]]]) # 更新历史股价序列
        X_past = np.append(X_past, np.log(S_past.iloc[-1] / S_past.iloc[-2])) # 更新历史股价日收益率序列

    return portfolio_values, government_values


g_aggressive = 0.5  # 激进的贪婪度参数
g_middle = 0.2
g_conservative = 0.05  # 保守的贪婪度参数

Vt_aggressive_1, V0 = trade_strategy(g_aggressive)
Vt_middle_1, _ = trade_strategy(g_middle)
Vt_conservative_1, _ = trade_strategy(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive_1, label= 'g_a = '+ str(g_aggressive))
plt.plot(days, Vt_middle_1, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative_1, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V0, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time (1 stock with M)')
plt.legend()
plt.show()


### 8 A Portfolio with Two Stocks
# p0 is around 0.38 in section 2

plt.figure(figsize=(12, 12))
plt.plot(S2_future[:500])
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price of 0066.HK")
plt.show()

def trade_strategy_2(g, r=0.001/100, p0 = 0.38):
    M = [100000]  # 初始资金
    S1 = [S1_past.iloc[-1]]  # 初始股票1价格
    S2 = [S2_past.iloc[-1]]  # 初始股票2价格
    N1 = [round(100000*p0/S1[0])]  # 初始股票1数量
    N2 = [round(100000*(1-p0)/S2[0])]  # 初始股票2数量
    
    portfolio_values = [100000]  # 投资组合价值列表
    government_values = [100000] # 国债价值变化列表
    trades = []  # 交易记录列表
    
    # 无需预测S2
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    
    S2_true_ = S2_future.copy()[:500]
    
    for t in range(0, len(S1_true_)):
        
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y_star(X1_past_, Y1_past)  # 预测下一天的分类
        today1 = digitize_X([X1_past_[-1]], epsilon) # 今天的分类
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # 今天的macd值
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # 今天对比昨天的差值
        
        act1 = action(today1,y1_star,macd1_new,delta1) # 决定买还是卖还是持有
        
        # 根据预测结果决定操作
        if act1 == 'Buy':
            shares_to_sell = round(g * N2[-1])  # 卖出N2的数量, 取整
            tent_money = shares_to_sell * S2[-1] # 中转价格
            shares_to_buy = round(tent_money / S1[-1]) # 买入N1的数量, 取整
            N1.append(N1[-1]+shares_to_buy)  # 更新N1(t)
            N2.append(N2[-1]-shares_to_sell)  # 更新N2(t)
            trades.append(('Buy', shares_to_buy, S1[-1]))  # 记录S1的交易
        if act1 == 'Sell':
            shares_to_sell = round(g * N1[-1])  # 卖出N1的数量, 取整
            tent_money = shares_to_sell * S1[-1] # 中转价格
            shares_to_buy = round(tent_money / S2[-1]) # 买入N2的数量, 取整
            N1.append(N1[-1]-shares_to_sell)  # 更新N1(t)
            N2.append(N2[-1]+shares_to_buy)  # 更新N2(t)
            trades.append(('Sell', shares_to_sell, S1[-1]))  # 记录S1的交易
        if act1 == 'Hold':
            trades.append(('Hold', 0, S1[-1]))  # 记录交易
        
        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S1.append(S1_true_.iloc[t])  # 更新实际股价S1(t)
        S2.append(S2_true_.iloc[t])  # 更新实际股价S2(t)
        portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # 更新投资组合价值
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # 更新历史股价S1序列
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # 更新历史股价日收益率X1序列
    
    return portfolio_values, government_values

g_aggressive = 0.5  # 激进的贪婪度参数
# g_middle = 0.2
g_conservative = 0.05  # 保守的贪婪度参数

Vt_aggressive_2, V0 = trade_strategy_2(g_aggressive)
# Vt_middle_2, _ = trade_strategy(g_middle)
Vt_conservative_2, _ = trade_strategy_2(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive_2, label= 'g_a = '+ str(g_aggressive))
# plt.plot(days, Vt_middle_2, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative_2, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V0, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time (2 stocks)')
plt.legend()
plt.show()
# 激进的策略更好，可能原因是S2一直下跌，太保守就会一直套牢

# 为了符合题意更为平稳的股票,我们把它设置成16.4附近波动,上涨看看
S2_future = pd.Series(index=S2_future.index)
previous_value =16.4  # 初始值
np.random.seed(17)
for date in S2_future.index:
    random_value = np.random.uniform(previous_value - 0.2, previous_value + 0.2)
    random_value = max(12, min(20, random_value))  # 确保随机值在范围内
    S2_future[date] = random_value
    previous_value = random_value

plt.figure(figsize=(12, 12))
plt.plot(S2_future[:500])
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price of 0066.HK")
plt.show()

Vt_aggressive_2, V0 = trade_strategy_2(g_aggressive)
# Vt_middle_2, _ = trade_strategy(g_middle)
Vt_conservative_2, _ = trade_strategy_2(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive_2, label= 'g_a = '+ str(g_aggressive))
# plt.plot(days, Vt_middle_2, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative_2, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V0, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time (2 stocks)')
plt.legend()
plt.show()

# 发现这个时候二者差不多

## 8.1 Efficient Frontier
# 还原S2_future
S2_future = S2[int(len(S2)/4*3)+1:]

def trade_strategy_2_1(g, r=0.001/100, p0 = 0.38):
    M = [100000]  # 初始资金
    S1 = [S1_past.iloc[-1]]  # 初始股票1价格
    S2 = [S2_past.iloc[-1]]  # 初始股票2价格
    N1 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0))]  # 初始股票1数量
    N2 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)]  # 初始股票2数量
    P0t= [p0]
    Pt = [p0]
    
    portfolio_values = [100000]  # 投资组合价值列表
    government_values = [100000] # 国债价值变化列表
    trades = []  # 交易记录列表
    
    # 无需预测S2
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    
    S2_true_ = S2_future.copy()[:500]
    S2_past_ = S2_past.copy()
    X2_past_ = X2_past.copy()
    
    for t in range(0, len(S1_true_)):
        
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y_star(X1_past_, Y1_past)  # 预测下一天的分类
        today1 = digitize_X([X1_past_[-1]], epsilon) # 今天的分类
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # 今天的macd值
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # 今天对比昨天的差值
        
        act1 = action(today1,y1_star,macd1_new,delta1) # 决定买还是卖还是持有
        
        # 根据预测结果决定操作
        if act1 == 'Buy':
            shares_to_sell = round(g * N2[-1])  # 卖出N2的数量, 取整
            tent_money = shares_to_sell * S2[-1] # 中转价格
            shares_to_buy = round(tent_money / S1[-1]) # 买入N1的数量, 取整
            N1.append(N1[-1]+shares_to_buy)  # 更新N1(t)
            N2.append(N2[-1]-shares_to_sell)  # 更新N2(t)
            trades.append(('Buy', shares_to_buy, S1[-1]))  # 记录S1的交易
        if act1 == 'Sell':
            shares_to_sell = round(g * N1[-1])  # 卖出N1的数量, 取整
            tent_money = shares_to_sell * S1[-1] # 中转价格
            shares_to_buy = round(tent_money / S2[-1]) # 买入N2的数量, 取整
            N1.append(N1[-1]-shares_to_sell)  # 更新N1(t)
            N2.append(N2[-1]+shares_to_buy)  # 更新N2(t)
            trades.append(('Sell', shares_to_sell, S1[-1]))  # 记录S1的交易
        if act1 == 'Hold':
            trades.append(('Hold', 0, S1[-1]))  # 记录交易
        
        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S1.append(S1_true_.iloc[t])  # 更新实际股价S1(t)
        S2.append(S2_true_.iloc[t])  # 更新实际股价S2(t)
        portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # 更新投资组合价值
        Pt.append(N1[-1]*S1[-1]/portfolio_values[-1]) # 更新Pt
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # 更新历史股价S1序列
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # 更新历史股价日收益率X1序列
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # 更新历史股价S2序列
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # 更新历史股价日收益率X2序列
        
        X_returns = pd.concat([pd.Series(X1_past_, name='港铁'),pd.Series(X2_past_, name='载通')],axis=1)
        mean_return = X_returns.mean()
        var_return = X_returns.var()
        cov_return =  X_returns.cov()
        p0new = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
        P0t.append(p0new)  # 更新P0
    
    return P0t, Pt

P0t, Pt = trade_strategy_2_1(0.2)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, P0t, label= 'P0t')
plt.plot(days, Pt, label= 'Pt')
plt.xlabel('Days')
plt.ylabel('Value')
plt.title('P and P0 over Time (g=0.2)')
plt.legend()
plt.show()

# Not too much.

## Next step
import sympy as sp
from sympy import *

def trade_strategy_2_2(g, r=0.001/100, p0 = 0.38):
    M = [100000]  # 初始资金
    S1 = [S1_past.iloc[-1]]  # 初始股票1价格
    S2 = [S2_past.iloc[-1]]  # 初始股票2价格
    N1 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0))]  # 初始股票1数量
    N2 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)]  # 初始股票2数量
    P0t= [p0]
    Pt = [p0]
    result = []
    
    portfolio_values = [100000]  # 投资组合价值列表
    government_values = [100000] # 国债价值变化列表
    trades = []  # 交易记录列表
    
    # 无需预测S2
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    
    S2_true_ = S2_future.copy()[:500]
    S2_past_ = S2_past.copy()
    X2_past_ = X2_past.copy()

    X_returns = pd.concat([pd.Series(X1_past_, name='港铁'),pd.Series(X2_past_, name='载通')],axis=1)
    mean_return = [X_returns.mean()]
    var_return = [X_returns.var()]
    cov_return =  [X_returns.cov()]
    sigma0 = p0**2*var_return[-1].iloc[0]+(1-p0)**2*var_return[-1].iloc[1]+2*p0*(1-p0)*cov_return[-1].iloc[0,1]
    sigma1 = X1_past_.var()
    sigmat = [sigma0]

    # 先进行一天
    Y1_past = digitize_X(X1_past_, epsilon)
    
    y1_star = compute_y_star(X1_past_, Y1_past)  # 预测下一天的分类
    today1 = digitize_X([X1_past_[-1]], epsilon) # 今天的分类
    _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
    macd1_new = macd1.iloc[-1] # 今天的macd值
    delta1 = macd1.iloc[-1]-macd1.iloc[-2] # 今天对比昨天的差值
    
    act1 = action(today1,y1_star,macd1_new,delta1) # 决定买还是卖还是持有
    
    # 根据预测结果决定操作
    if act1 == 'Buy':
        shares_to_sell = round(g * N2[-1])  # 卖出N2的数量, 取整
        tent_money = shares_to_sell * S2[-1] # 中转价格
        shares_to_buy = round(tent_money / S1[-1]) # 买入N1的数量, 取整
        N1.append(N1[-1]+shares_to_buy)  # 更新N1(t)
        N2.append(N2[-1]-shares_to_sell)  # 更新N2(t)
        trades.append(('Buy', shares_to_buy, S1[-1]))  # 记录S1的交易
    if act1 == 'Sell':
        shares_to_sell = round(g * N1[-1])  # 卖出N1的数量, 取整
        tent_money = shares_to_sell * S1[-1] # 中转价格
        shares_to_buy = round(tent_money / S2[-1]) # 买入N2的数量, 取整
        N1.append(N1[-1]-shares_to_sell)  # 更新N1(t)
        N2.append(N2[-1]+shares_to_buy)  # 更新N2(t)
        trades.append(('Sell', shares_to_sell, S1[-1]))  # 记录S1的交易
    if act1 == 'Hold':
        trades.append(('Hold', 0, S1[-1]))  # 记录交易
    
    government_values.append(100000 * (1 + r) ** 0)  # 更新国债价值
    S1.append(S1_true_.iloc[0])  # 更新实际股价S1(t)
    S2.append(S2_true_.iloc[0])  # 更新实际股价S2(t)
    portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # 更新投资组合价值
    Pt.append(N1[-1]*S1[-1]/portfolio_values[-1]) # 更新Pt
    
    S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[0]]]) # 更新历史股价S1序列
    X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # 更新历史股价日收益率X1序列
    S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[0]]]) # 更新历史股价S2序列
    X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # 更新历史股价日收益率X2序列
    
    X_returns = pd.concat([pd.Series(X1_past_, name='港铁'),pd.Series(X2_past_, name='载通')],axis=1)
    mean_return.append(X_returns.mean())
    var_return.append(X_returns.var())
    cov_return.append(X_returns.cov())
    p0new = (var_return[-1].iloc[1]-cov_return[-1].iloc[0,1])/(var_return[-1].iloc[0]+var_return[-1].iloc[1]-2*cov_return[-1].iloc[0,1])
    P0t.append(p0new)  # 更新P0
    
    sigmat.append(Pt[-1]**2*var_return[-1].iloc[0]+(1-Pt[-1])**2*var_return[-1].iloc[1]+2*Pt[-1]*(1-Pt[-1])*cov_return[-1].iloc[0,1]) # 更新sigmat

    # 后续根据sigmat选择ni
    for t in range(1, len(S1_true_)):
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y_star(X1_past_, Y1_past)  # 预测下一天的分类
        today1 = digitize_X([X1_past_[-1]], epsilon) # 今天的分类
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # 今天的macd值
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # 今天对比昨天的差值
        
        act1 = action(today1,y1_star,macd1_new,delta1) # 决定买还是卖还是持有
        
        if act1 == 'Buy':
            sig = sigmat[-1]+g*(sigma1-sigmat[-1])
        if act1 == 'Sell':
            sig = sigmat[-1]-g*(sigmat[-1]-sigma0)
        
        a = sp.Symbol('a')
        # 定义方程
        equation = sp.Eq(sig, a**2 * var_return[-1].iloc[0] + (1 - a)**2 * var_return[-1].iloc[1] + 2 * a * (1 - a) * cov_return[-1].iloc[0,1])
        # 解方程
        solutions = sp.solve(equation, a)
        filter_ = [x for x in solutions if (not isinstance(x, sp.core.mul.Mul) and not isinstance(x, sp.core.add.Add))]
        filter_ = [x for x in filter_ if P0t[-1] <= x <= 1]

        if len(filter_)!=0:
            result.append(filter_)
        
        finalN1 = round(result[-1][-1]*portfolio_values[-1]/S1[-1])
        finalN2 = round((1-result[-1][-1])*portfolio_values[-1]/S2[-1])

        # 根据预测结果决定操作
        if N1[-1] < finalN1:
            shares_to_buy = finalN1 - N1[-1] # 买入N1的数量, 取整
            N1.append(finalN1)  # 更新N1(t)
            N2.append(finalN2)  # 更新N2(t)
            trades.append(('Buy', shares_to_buy, S1[-1]))  # 记录S1的交易
        if N1[-1] > finalN1:
            shares_to_sell = N1[-1] - finalN1  # 卖出N1的数量, 取整
            N1.append(finalN1)  # 更新N1(t)
            N2.append(finalN2)  # 更新N2(t)
            trades.append(('Sell', shares_to_sell, S1[-1]))  # 记录S1的交易
        if act1 == 'Hold':
            trades.append(('Hold', 0, S1[-1]))  # 记录交易

        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S1.append(S1_true_.iloc[t])  # 更新实际股价S1(t)
        S2.append(S2_true_.iloc[t])  # 更新实际股价S2(t)
        portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # 更新投资组合价值
        Pt.append(result[-1][-1]) # 更新Pt

        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # 更新历史股价S1序列
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # 更新历史股价日收益率X1序列
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # 更新历史股价S2序列
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # 更新历史股价日收益率X2序列

        X_returns = pd.concat([pd.Series(X1_past_, name='港铁'),pd.Series(X2_past_, name='载通')],axis=1)
        mean_return.append(X_returns.mean())
        var_return.append(X_returns.var())
        cov_return.append(X_returns.cov())
        p0new = (var_return[-1].iloc[1]-cov_return[-1].iloc[0,1])/(var_return[-1].iloc[0]+var_return[-1].iloc[1]-2*cov_return[-1].iloc[0,1])
        P0t.append(p0new)  # 更新P0

        sigmat.append(Pt[-1]**2*var_return[-1].iloc[0]+(1-Pt[-1])**2*var_return[-1].iloc[1]+2*Pt[-1]*(1-Pt[-1])*cov_return[-1].iloc[0,1]) # 更新sigmat

    return portfolio_values, government_values


Vt_aggressive, V0 = trade_strategy_2_2(g_aggressive)
# Vt_middle, _ = trade_strategy(g_middle)
Vt_conservative, _ = trade_strategy_2_2(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive, label= 'g_a = '+ str(g_aggressive))
# plt.plot(days, Vt_middle, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V0, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time (2 stocks optimized)')
plt.legend()
plt.show()

# 确实稳健了很多


### 9 A Portfolio with Two Stocks and Money
epsilon = 0.004

# X to Y
Y1_digi = digitize_X(X1_past, epsilon)
Y2_digi = digitize_X(X2_past, epsilon)

## 绘制 conditional PDF,选用正态假设
def conditional_pdf(X, Y, x, y):
    filtered_X = X[:-1][Y[1:] == y]
    return norm.pdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X))
    # return logistic.pdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X))


prior_y1D = len(Y1_digi[Y1_digi=='D'])/len(Y1_digi)
prior_y1H = len(Y1_digi[Y1_digi=='H'])/len(Y1_digi)
prior_y1U = len(Y1_digi[Y1_digi=='U'])/len(Y1_digi)
priors1 = {'D':prior_y1D,'H':prior_y1H, 'U':prior_y1U}

prior_y2D = len(Y2_digi[Y2_digi=='D'])/len(Y2_digi)
prior_y2H = len(Y2_digi[Y2_digi=='H'])/len(Y2_digi)
prior_y2U = len(Y2_digi[Y2_digi=='U'])/len(Y2_digi)
priors2 = {'D':prior_y2D,'H':prior_y2H, 'U':prior_y2U}

def compute_y1_star(X, Y):
    x = X[-1]  # 观测值 X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors1[y] * conditional_pdf(X, Y, x, y)  # 计算 q(y) * f_y(x)
        if prob > max_prob:
            max_prob = prob
            y_star = y
    return y_star

def compute_y2_star(X, Y):
    x = X[-1]  # 观测值 X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors2[y] * conditional_pdf(X, Y, x, y)  # 计算 q(y) * f_y(x)
        if prob > max_prob:
            max_prob = prob
            y_star = y
    return y_star

# y1_star = compute_y1_star(X1_past, Y1_digi)
# y2_star = compute_y2_star(X2_past, Y2_digi)


def trade_strategy_3(g, r=0.001/100,p0 = 0.38):
    M = [100000]  # 初始资金
    N1 = [0]  # 初始股票1数量
    N2 = [0]  # 初始股票2数量
    S1 = [S1_past.iloc[-1]]  # 初始股票价格
    S2 = [S2_past.iloc[-1]]  # 初始股票价格
    
    portfolio_values = [100000]  # 投资组合价值列表
    government_values = [100000] # 国债价值变化列表
    trades = []  # 交易记录列表
    P0t= [p0]
    Pt = [p0]
    
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    S2_past_ = S2_past.copy()
    S2_true_ = S2_future.copy()[:500]
    X2_past_ = X2_past.copy()
    
    for t in range(0, len(S1_true_)):
        
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y1_star(X1_past_, Y1_past)  # 预测下一天的分类
        today1 = digitize_X([X1_past_[-1]], epsilon) # 今天的分类
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # 今天的macd值
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # 今天对比昨天的差值
        
        act1 = action(today1,y1_star,macd1_new,delta1) # 决定股票1买还是卖还是持有
        
        Y2_past = digitize_X(X2_past_, epsilon)
        
        y2_star = compute_y2_star(X2_past_, Y2_past)  # 预测下一天的分类
        today2 = digitize_X([X2_past_[-1]], epsilon) # 今天的分类
        _, _, macd2 = talib.MACD(S2_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd2_new = macd2.iloc[-1] # 今天的macd值
        delta2 = macd2.iloc[-1]-macd2.iloc[-2] # 今天对比昨天的差值
        
        act2 = action(today2,y2_star,macd2_new,delta2) # 决定股票2买还是卖还是持有        
        
        # 根据预测结果决定操作
        if act1 == 'Buy' and act2 == 'Buy': # 按照g平均分配M购买1和2
            n1_buy = round(g * M[-1]/2 / S1[-1])  # 购买的股票1数量, 取整
            n2_buy = round(g * M[-1]/2 / S2[-1])  # 购买的股票2数量, 取整
            if (N1[-1]+n1_buy)*S1[-1]/((N1[-1]+n1_buy)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*p0/S1[-1])
                n2_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*(1-p0)/S2[-1])
                n1_buy = n1_end - N1[-1]
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]+n1_buy)  # 更新N1(t)
            N2.append(N2[-1]+n2_buy)  # 更新N2(t)
            M.append(M[-1]-n1_buy*S1[-1]-n2_buy*S2[-1]) # 更新M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易
        
        if act1 == 'Buy' and act2 == 'Hold': # 按照g分配M购买1
            n1_buy = round(g * M[-1] / S1[-1])  # 购买的股票1数量, 取整
            N1.append(N1[-1]+n1_buy)  # 更新N1(t)
            N2.append(N2[-1])  # 更新N2(t)
            M.append(M[-1]-n1_buy*S1[-1]) # 更新M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        
        if act1 == 'Buy' and act2 == 'Sell': # 先卖2,加上按照g分配的M购买1
            n2_sell = round(g * N2[-1])  # 卖出的股票2数量, 取整
            n1_buy = round(g * (M[-1] + n2_sell*S2[-1]) / S1[-1])  # 购买的股票1数量, 取整
            N1.append(N1[-1]+n1_buy)  # 更新N1(t)
            N2.append(N2[-1]-n2_sell)  # 更新N2(t)
            M.append(M[-1]-n1_buy*S1[-1]+n2_sell*S2[-1]) # 更新M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易        
        
        if act1 == 'Hold' and act2 == 'Buy': # 按照g分配M购买2
            n2_buy = round(g * M[-1] / S2[-1])  # 购买的股票2数量, 取整
            if N1[-1]*S1[-1]/(N1[-1]*S1[-1]+(N2[-1]+n2_buy)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n2_end = round(N1[-1]*S1[-1]*(1-p0)/p0/S2[-1])
                n2_buy = max(n2_end - N2[-1],0)
            N1.append(N1[-1])  # 更新N1(t)
            N2.append(N2[-1]+n2_buy)  # 更新N2(t)
            M.append(M[-1]-n2_buy*S2[-1]) # 更新M(t)
            trades.append(('Hold1', 0, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易        
        
        if act1 == 'Hold' and act2 == 'Hold': # 不变
            N1.append(N1[-1])  # 更新N1(t)
            N2.append(N2[-1])  # 更新N2(t)
            M.append(M[-1]) # 更新M(t)
            trades.append(('Hold1', 0, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        
        if act1 == 'Hold' and act2 == 'Sell': # 按照g分配N2卖出2
            n2_sell = round(g * N2[-1])  # 购买的股票2数量, 取整
            N1.append(N1[-1])  # 更新N1(t)
            N2.append(N2[-1]-n2_sell)  # 更新N2(t)
            M.append(M[-1]+n2_sell*S2[-1]) # 更新M(t)
            trades.append(('Hold1', 0, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易  
        
        if act1 == 'Sell' and act2 == 'Buy': # 先卖2,加上按照g分配的M购买1
            n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
            n2_buy = round(g * (M[-1] + n1_sell*S1[-1]) / S2[-1])  # 购买的股票2数量, 取整
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*p0/S1[-1])
                n2_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]-n1_sell)  # 更新N1(t)
            N2.append(N2[-1]+n2_buy)  # 更新N2(t)
            M.append(M[-1]+n1_sell*S1[-1]-n2_buy*S2[-1]) # 更新M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易
        
        if act1 == 'Sell' and act2 == 'Hold': # 按照g分配N1卖出1
            n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1]+N2[-1]*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round(N2[-1]*S2[-1]*p0/(1-p0)/S1[-1])
                n1_sell = max(N1[-1]-n1_end,0)
            N1.append(N1[-1]-n1_sell)  # 更新N1(t)
            N2.append(N2[-1])  # 更新N2(t)
            M.append(M[-1]+n1_sell*S1[-1]) # 更新M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        
        if act1 == 'Sell' and act2 == 'Sell': # 按照g平均分配N1和N2卖出
            n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
            n2_sell = round(g * N2[-1])  # 卖出的股票2数量, 取整
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]-n2_sell)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*p0/S1[-1])
                n2_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_sell = N2[-1] - n2_end
            N1.append(N1[-1]-n1_sell)  # 更新N1(t)
            N2.append(N2[-1]-n2_sell)  # 更新N2(t)
            M.append(M[-1]+n1_sell*S1[-1]+n2_sell*S2[-1]) # 更新M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易
        
        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S1.append(S1_true_.iloc[t])  # 更新实际股价S1(t)
        S2.append(S2_true_.iloc[t])  # 更新实际股价S2(t)
        portfolio_values.append(M[-1]+N1[-1]*S1[-1]+N2[-1]*S2[-1])  # 更新投资组合价值
        Pt.append(N1[-1]*S1[-1]/(N1[-1]*S1[-1]+N2[-1]*S2[-1])) # 更新Pt
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # 更新历史股价S1序列
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # 更新历史股价日收益率X1序列
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # 更新历史股价S2序列
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # 更新历史股价日收益率X2序列
        X_returns = pd.concat([pd.Series(X1_past_, name='港铁'),pd.Series(X2_past_, name='载通')],axis=1)
        
        mean_return = X_returns.mean()
        var_return = X_returns.var()
        cov_return =  X_returns.cov()
        p0new = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
        P0t.append(p0new)  # 更新P0

    return portfolio_values, government_values


g_aggressive = 0.5  # 激进的贪婪度参数
g_middle = 0.2
g_conservative = 0.05  # 保守的贪婪度参数

Vt_aggressive_3, V0 = trade_strategy_3(g_aggressive)
Vt_middle_3, _ = trade_strategy_3(g_middle)
Vt_conservative_3, _ = trade_strategy_3(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive_3, label= 'g_a = '+ str(g_aggressive))
plt.plot(days, Vt_middle_3, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative_3, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V0, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time (2 stocks with M)')
plt.legend()
plt.show()

# 纯依托答辩


### 引入RSI指标 相对强弱指数 超卖则买 超买则卖 先考虑独立情况实验
def rsi_indicator(price_data, window=14):
    rsi = talib.RSI(price_data, timeperiod=window)
    return rsi
    
def action2(today,y_star,macd_new,delta,rsi):
    if (today == 'U' and y_star == 'U' and macd_new > 0 and delta > 0 and rsi<30) or (today == 'H' and y_star == 'U' and rsi<30) or (today == 'D' and (y_star == 'U' or y_star == 'H' and rsi<30)):
        act = 'Buy'
    elif (today == 'U' and y_star == 'U' and macd_new < 0 and delta < 0 and rsi>70) or (today == 'U' and (y_star == 'H' or y_star == 'D') and rsi>70) or (today == 'H' and y_star == 'D' and rsi>70) or (today == 'D' and y_star == 'D' and macd_new < 0 and rsi>70):
        act = 'Sell'
    else:
        act = 'Hold'
    return act

def trade_strategy_4(g,a=0.65, r=0.001/100):#这里设置的a是分配给两只股票资金的比例，比如设置a=0.5，就是两只股票最多动用的资金就是5万
    M = [100000]  # 初始资金
    N_1 = [0]  # S1初始股票数量
    N_2 = [0]  #S2初始股票数量    
    S_1 = [S1_past.iloc[-1]]  # S1初始股票价格
    S_2= [S2_past.iloc[-1]] #S2初始股票价格
    M_1=[elem * a for elem in M]
    M_2=[elem * (1-a) for elem in M]
    portfolio_values = [100000]  # 投资组合价值列表
    government_values = [100000] # 国债价值变化列表
    trades_S1 = [] # S1交易记录列表
    trades_S2 = [] # S2交易记录列表
    
    S1_pas = S1_past.copy()
    S2_pas=S2_past.copy()
    S1_true = S1_future.copy()[:500]
    S2_true=S2_future.copy()[:500]
    X1_pas = X1_past.copy()
    X2_pas=X2_past.copy()
    
    for t in range(0, len(S1_true)):
        
        Y1_past = digitize_X(X1_pas, epsilon)
        Y2_past=digitize_X(X2_pas,epsilon)
        y_star_S1 = compute_y1_star(X1_pas, Y1_past)  # 股票S1预测下一天的分类
        y_star_S2 = compute_y2_star(X2_pas, Y2_past)  # 股票S2预测下一天的分类
        today_S1 = digitize_X([X1_pas[-1]], epsilon) # 股票S1今天的分类
        today_S2 = digitize_X([X2_pas[-1]], epsilon)#股票S2今天的分类
        _, _, macd_S1 = talib.MACD(S1_pas, fastperiod=12, slowperiod=26, signalperiod=9)
        _, _, macd_S2 = talib.MACD(S2_pas, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_new_S1 = macd_S1.iloc[-1] # 今天的macd值
        macd_new_S2 =macd_S2.iloc[-1]
        delta_S1 = macd_S1.iloc[-1]-macd_S1.iloc[-2] # 今天对比昨天的差值
        delta_S2 = macd_S2.iloc[-1]-macd_S2.iloc[-2]
        rsi_S1=rsi_indicator(S1_pas[-15:], window=14).iloc[-1] #考虑在交易日那天股票S1的RSI值，如果值>70,就卖，<30就买，30-70之间就不动
        rsi_S2=rsi_indicator(S2_pas[-15:], window=14).iloc[-1] #考虑在交易日那天股票S2的RSI值，如果值>70,就卖，<30就买，30-70之间就不动
        act_S1 = action2(today_S1,y_star_S1,macd_new_S1,delta_S1,rsi_S1) # 对于股票S1决定买还是卖还是持有
        act_S2=action2(today_S2,y_star_S2,macd_new_S2,delta_S2,rsi_S2) # 对于股票S2决定买还是卖还是持有
        
        # 根据预测结果决定操作
        if act_S1 == 'Buy':
            shares_to_buy = round(g * M_1[-1] / S_1[-1])  # 购买的股票数量, 取整
            N_1.append(N_1[-1]+shares_to_buy)  # 更新N(t)
            M_1.append(M_1[-1]-shares_to_buy*S_1[-1]) # 更新M(t)
            trades_S1.append(('Buy', shares_to_buy, S_1[-1]))  # 记录交易
        
        if act_S1 == 'Sell':
            shares_to_sell = round(g * N_1[-1])  # 卖出股票的数量, 取整
            N_1.append(N_1[-1]-shares_to_sell)  # 更新N(t)
            M_1.append(M_1[-1]+shares_to_sell*S_1[-1]) # 更新M(t)
            trades_S1.append(('Sell', shares_to_sell, S_1[-1]))  # 记录交易
        if act_S1 == 'Hold':
            trades_S1.append(('Hold', 0, S_1[-1]))  # 记录交易

        if act_S2 == 'Buy':
            shares_to_buy = round(g * M_2[-1] / S_2[-1])  # 购买的股票数量, 取整
            N_2.append(N_2[-1]+shares_to_buy)  # 更新N(t)
            M_2.append(M_2[-1]-shares_to_buy*S_2[-1]) # 更新M(t)
            trades_S2.append(('Buy', shares_to_buy, S_2[-1]))  # 记录交易
        
        if act_S2 == 'Sell':
            shares_to_sell = round(g * N_2[-1])  # 卖出股票的数量, 取整
            N_2.append(N_2[-1]-shares_to_sell)  # 更新N(t)
            M_2.append(M_2[-1]+shares_to_sell*S_2[-1]) # 更新M(t)
            trades_S2.append(('Sell', shares_to_sell, S_2[-1]))  # 记录交易
        if act_S2 == 'Hold':
            trades_S2.append(('Hold', 0, S_2[-1]))  # 记录交易
        
        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S_1.append(S1_true.iloc[t])  # 更新实际股价S1(t)
        S_2.append(S2_true.iloc[t])  # 更新实际股价S1(t)
        portfolio_values.append(M_1[-1]+N_1[-1]*S_1[-1]+M_2[-1]+N_2[-1]*S_2[-1])  # 更新投资组合价值
        
        S1_pas = pd.concat([S1_pas, S1_true.iloc[[t]]]) # 更新S1历史股价序列
        S2_pas= pd.concat([S2_pas, S2_true.iloc[[t]]]) #更新S2历史股价序列
        X1_pas = np.append(X1_pas, np.log(S1_pas.iloc[-1] / S1_pas.iloc[-2])) # 更新历史股价日收益率序列
        X2_pas = np.append(X2_pas, np.log(S2_pas.iloc[-1] / S2_pas.iloc[-2]))

    return portfolio_values, government_values
g_aggressive = 0.5  # 激进的贪婪度参数
g_middle = 0.2
g_conservative = 0.05 # 保守的贪婪度参数
Vt_aggressive_4, V04 = trade_strategy_4(g_aggressive)
Vt_middle_4, _ = trade_strategy_4(g_middle)
Vt_conservative_4, _ = trade_strategy_4(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive_4, label= 'g_a = '+ str(g_aggressive))
plt.plot(days, Vt_middle_4, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative_4, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V04, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time (2 stocks with M optimized)')
plt.legend()
plt.show()


### 综合section7, section8, RSI index
def trade_strategy_5(g, r=0.001/100,p0 = 0.38 ,a=0.65):
    M = [100000]  # 初始资金
    N1 = [0]  # 初始股票1数量
    N2 = [0]  # 初始股票2数量
    S1 = [S1_past.iloc[-1]]  # 初始股票价格
    S2 = [S2_past.iloc[-1]]  # 初始股票价格
    
    portfolio_values = [100000]  # 投资组合价值列表
    government_values = [100000] # 国债价值变化列表
    trades = []  # 交易记录列表
    P0t= [p0]
    Pt = [p0]
    
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    S2_past_ = S2_past.copy()
    S2_true_ = S2_future.copy()[:500]
    X2_past_ = X2_past.copy()
    
    for t in range(0, len(S1_true_)):
        
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y1_star(X1_past_, Y1_past)  # 预测下一天的分类
        today1 = digitize_X([X1_past_[-1]], epsilon) # 今天的分类
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # 今天的macd值
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # 今天对比昨天的差值
        
        Y2_past = digitize_X(X2_past_, epsilon)
        
        y2_star = compute_y2_star(X2_past_, Y2_past)  # 预测下一天的分类
        today2 = digitize_X([X2_past_[-1]], epsilon) # 今天的分类
        _, _, macd2 = talib.MACD(S2_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd2_new = macd2.iloc[-1] # 今天的macd值
        delta2 = macd2.iloc[-1]-macd2.iloc[-2] # 今天对比昨天的差值
        
        rsi_S1=rsi_indicator(S1_past_[-15:], window=14).iloc[-1] #考虑在交易日那天股票S1的RSI值，如果值>70,就卖，<30就买，30-70之间就不动
        rsi_S2=rsi_indicator(S2_past_[-15:], window=14).iloc[-1] #考虑在交易日那天股票S2的RSI值，如果值>70,就卖，<30就买，30-70之间就不动
        
        
        act1 = action2(today1,y1_star,macd1_new,delta1,rsi_S1) # 对于股票S1决定买还是卖还是持有
        act2 = action2(today2,y2_star,macd2_new,delta2,rsi_S2) # 对于股票S2决定买还是卖还是持有
        
        # act1 = action(today1,y1_star,macd1_new,delta1) # 决定股票1买还是卖还是持有
        # act2 = action(today2,y2_star,macd2_new,delta2) # 决定股票2买还是卖还是持有   
        
        # 根据预测结果决定操作
        if act1 == 'Buy' and act2 == 'Buy': # 按照g平均分配M购买1和2
            n1_buy = round(g * M[-1]/2 / S1[-1])  # 购买的股票1数量, 取整
            n2_buy = round(g * M[-1]/2 / S2[-1])  # 购买的股票2数量, 取整
            if (N1[-1]+n1_buy)*S1[-1]/((N1[-1]+n1_buy)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*p0/S1[-1])
                n2_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*(1-p0)/S2[-1])
                n1_buy = n1_end - N1[-1]
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]+n1_buy)  # 更新N1(t)
            N2.append(N2[-1]+n2_buy)  # 更新N2(t)
            M.append(M[-1]-n1_buy*S1[-1]-n2_buy*S2[-1]) # 更新M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易
        
        if act1 == 'Buy' and act2 == 'Hold': # 按照g分配M购买1
            n1_buy = round(g * M[-1] / S1[-1])  # 购买的股票1数量, 取整
            N1.append(N1[-1]+n1_buy)  # 更新N1(t)
            N2.append(N2[-1])  # 更新N2(t)
            M.append(M[-1]-n1_buy*S1[-1]) # 更新M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        
        if act1 == 'Buy' and act2 == 'Sell': # 先卖2,加上按照g分配的M购买1
            n2_sell = round(g * N2[-1])  # 卖出的股票2数量, 取整
            n1_buy = round(g * (M[-1] + n2_sell*S2[-1]) / S1[-1])  # 购买的股票1数量, 取整
            N1.append(N1[-1]+n1_buy)  # 更新N1(t)
            N2.append(N2[-1]-n2_sell)  # 更新N2(t)
            M.append(M[-1]-n1_buy*S1[-1]+n2_sell*S2[-1]) # 更新M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易        
        
        if act1 == 'Hold' and act2 == 'Buy': # 按照g分配M购买2
            n2_buy = round(g * M[-1] / S2[-1])  # 购买的股票2数量, 取整
            if N1[-1]*S1[-1]/(N1[-1]*S1[-1]+(N2[-1]+n2_buy)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n2_end = round(N1[-1]*S1[-1]*(1-p0)/p0/S2[-1])
                n2_buy = max(n2_end - N2[-1],0)
            N1.append(N1[-1])  # 更新N1(t)
            N2.append(N2[-1]+n2_buy)  # 更新N2(t)
            M.append(M[-1]-n2_buy*S2[-1]) # 更新M(t)
            trades.append(('Hold1', 0, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易        
        
        if act1 == 'Hold' and act2 == 'Hold': # 不变
            N1.append(N1[-1])  # 更新N1(t)
            N2.append(N2[-1])  # 更新N2(t)
            M.append(M[-1]) # 更新M(t)
            trades.append(('Hold1', 0, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        
        if act1 == 'Hold' and act2 == 'Sell': # 按照g分配N2卖出2
            n2_sell = round(g * N2[-1])  # 购买的股票2数量, 取整
            N1.append(N1[-1])  # 更新N1(t)
            N2.append(N2[-1]-n2_sell)  # 更新N2(t)
            M.append(M[-1]+n2_sell*S2[-1]) # 更新M(t)
            trades.append(('Hold1', 0, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易  
        
        if act1 == 'Sell' and act2 == 'Buy': # 先卖2,加上按照g分配的M购买1
            n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
            n2_buy = round(g * (M[-1] + n1_sell*S1[-1]) / S2[-1])  # 购买的股票2数量, 取整
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*p0/S1[-1])
                n2_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]-n1_sell)  # 更新N1(t)
            N2.append(N2[-1]+n2_buy)  # 更新N2(t)
            M.append(M[-1]+n1_sell*S1[-1]-n2_buy*S2[-1]) # 更新M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易
        
        if act1 == 'Sell' and act2 == 'Hold': # 按照g分配N1卖出1
            n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1]+N2[-1]*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round(N2[-1]*S2[-1]*p0/(1-p0)/S1[-1])
                n1_sell = max(N1[-1]-n1_end,0)
            N1.append(N1[-1]-n1_sell)  # 更新N1(t)
            N2.append(N2[-1])  # 更新N2(t)
            M.append(M[-1]+n1_sell*S1[-1]) # 更新M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        
        if act1 == 'Sell' and act2 == 'Sell': # 按照g平均分配N1和N2卖出
            n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
            n2_sell = round(g * N2[-1])  # 卖出的股票2数量, 取整
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]-n2_sell)*S2[-1]) < p0:  # 调整pt比例不低于p0
                n1_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*p0/S1[-1])
                n2_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_sell = N2[-1] - n2_end
            N1.append(N1[-1]-n1_sell)  # 更新N1(t)
            N2.append(N2[-1]-n2_sell)  # 更新N2(t)
            M.append(M[-1]+n1_sell*S1[-1]+n2_sell*S2[-1]) # 更新M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易
        
        government_values.append(100000 * (1 + r) ** t)  # 更新国债价值
        S1.append(S1_true_.iloc[t])  # 更新实际股价S1(t)
        S2.append(S2_true_.iloc[t])  # 更新实际股价S2(t)
        portfolio_values.append(M[-1]+N1[-1]*S1[-1]+N2[-1]*S2[-1])  # 更新投资组合价值
        # Pt.append(N1[-1]*S1[-1]/(N1[-1]*S1[-1]+N2[-1]*S2[-1])) # 更新Pt
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # 更新历史股价S1序列
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # 更新历史股价日收益率X1序列
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # 更新历史股价S2序列
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # 更新历史股价日收益率X2序列
        X_returns = pd.concat([pd.Series(X1_past_, name='港铁'),pd.Series(X2_past_, name='载通')],axis=1)
        
        mean_return = X_returns.mean()
        var_return = X_returns.var()
        cov_return =  X_returns.cov()
        p0new = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
        P0t.append(p0new)  # 更新P0

    return portfolio_values, government_values


g_aggressive = 0.5  # 激进的贪婪度参数
g_middle = 0.2
g_conservative = 0.05  # 保守的贪婪度参数

Vt_aggressive_5, V0 = trade_strategy_5(g_aggressive)
Vt_middle_5, _ = trade_strategy_5(g_middle)
Vt_conservative_5, _ = trade_strategy_5(g_conservative)

days = range(501)

plt.figure(figsize=(12, 12))
plt.plot(days, Vt_aggressive_5, label= 'g_a = '+ str(g_aggressive))
plt.plot(days, Vt_middle_5, label= 'g_m = '+ str(g_middle))
plt.plot(days, Vt_conservative_5, label= 'g_c = ' + str(g_conservative))
plt.plot(days, V0, label= 'government')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time (2 stocks with M ultra-optimized)')
plt.legend()
plt.show()

### 2024.5.5 更新: 下面的是错误代码，计算的是N1/N2

        # # 根据预测结果决定操作
        # if act1 == 'Buy' and act2 == 'Buy': # 按照g平均分配M购买1和2
        #     n1_buy = round(g * M[-1]/2 / S1[-1])  # 购买的股票1数量, 取整
        #     n2_buy = round(g * M[-1]/2 / S2[-1])  # 购买的股票2数量, 取整
        #     if (N1[-1]+n1_buy)/(N1[-1]+n1_buy + N2[-1]+n2_buy) < p0:  # 调整N1/N2比例不低于p0
        #         n1_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0))
        #         n2_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)
        #         n1_buy = n1_end - N1[-1]
        #         n2_buy = n2_end - N2[-1]
        #     N1.append(N1[-1]+n1_buy)  # 更新N1(t)
        #     N2.append(N2[-1]+n2_buy)  # 更新N2(t)
        #     M.append(M[-1]-n1_buy*S1[-1]-n2_buy*S2[-1]) # 更新M(t)
        #     trades.append(('Buy1', n1_buy, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易
        # 
        # if act1 == 'Buy' and act2 == 'Hold': # 按照g分配M购买1
        #     n1_buy = round(g * M[-1] / S1[-1])  # 购买的股票1数量, 取整
        #     N1.append(N1[-1]+n1_buy)  # 更新N1(t)
        #     N2.append(N2[-1])  # 更新N2(t)
        #     M.append(M[-1]-n1_buy*S1[-1]) # 更新M(t)
        #     trades.append(('Buy1', n1_buy, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        # 
        # if act1 == 'Buy' and act2 == 'Sell': # 先卖2,加上按照g分配的M购买1
        #     n2_sell = round(g * N2[-1])  # 卖出的股票2数量, 取整
        #     n1_buy = round(g * (M[-1] + n2_sell*S2[-1]) / S1[-1])  # 购买的股票1数量, 取整
        #     N1.append(N1[-1]+n1_buy)  # 更新N1(t)
        #     N2.append(N2[-1]-n2_sell)  # 更新N2(t)
        #     M.append(M[-1]-n1_buy*S1[-1]+n2_sell*S2[-1]) # 更新M(t)
        #     trades.append(('Buy1', n1_buy, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易        
        # 
        # if act1 == 'Hold' and act2 == 'Buy': # 按照g分配M购买2
        #     n2_buy = round(g * M[-1] / S2[-1])  # 购买的股票2数量, 取整
        #     if N1[-1]/(N1[-1]+N2[-1]+n2_buy) < p0:  # 调整N1/N2比例不低于p0
        #         n2_end = round(N1[-1]*(1-p0)/p0)
        #         n2_buy = max(n2_end - N2[-1],0)
        #     N1.append(N1[-1])  # 更新N1(t)
        #     N2.append(N2[-1]+n2_buy)  # 更新N2(t)
        #     M.append(M[-1]-n2_buy*S2[-1]) # 更新M(t)
        #     trades.append(('Hold1', 0, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易        
        # 
        # if act1 == 'Hold' and act2 == 'Hold': # 不变
        #     N1.append(N1[-1])  # 更新N1(t)
        #     N2.append(N2[-1])  # 更新N2(t)
        #     M.append(M[-1]) # 更新M(t)
        #     trades.append(('Hold1', 0, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        # 
        # if act1 == 'Hold' and act2 == 'Sell': # 按照g分配N2卖出2
        #     n2_sell = round(g * N2[-1])  # 购买的股票2数量, 取整
        #     N1.append(N1[-1])  # 更新N1(t)
        #     N2.append(N2[-1]-n2_sell)  # 更新N2(t)
        #     M.append(M[-1]+n2_sell*S2[-1]) # 更新M(t)
        #     trades.append(('Hold1', 0, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易  
        # 
        # if act1 == 'Sell' and act2 == 'Buy': # 先卖2,加上按照g分配的M购买1
        #     n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
        #     n2_buy = round(g * (M[-1] + n1_sell*S1[-1]) / S2[-1])  # 购买的股票2数量, 取整
        #     if (N1[-1]-n1_sell)/(N1[-1]-n1_sell + N2[-1]+n2_buy) < p0:  # 调整N1/N2比例不低于p0
        #         n1_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))/(S1[-1]+S2[-1]*(1-p0)/p0))
        #         n2_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)
        #         n1_sell = N1[-1] - n1_end
        #         n2_buy = n2_end - N2[-1]
        #     N1.append(N1[-1]-n1_sell)  # 更新N1(t)
        #     N2.append(N2[-1]+n2_buy)  # 更新N2(t)
        #     M.append(M[-1]+n1_sell*S1[-1]-n2_buy*S2[-1]) # 更新M(t)
        #     trades.append(('Sell1', n1_sell, S1[-1],'Buy2',n2_buy, S2[-1]))  # 记录交易
        # 
        # if act1 == 'Sell' and act2 == 'Hold': # 按照g分配N1卖出1
        #     n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
        #     if (N1[-1]-n1_sell)/(N1[-1]+N2[-1]-n1_sell) < p0:  # 调整N1/N2比例不低于p0
        #         n1_end = round(N2[-1]*p0/(1-p0))
        #         n1_sell = max(N1[-1]-n1_end,0)
        #     N1.append(N1[-1]-n1_sell)  # 更新N1(t)
        #     N2.append(N2[-1])  # 更新N2(t)
        #     M.append(M[-1]+n1_sell*S1[-1]) # 更新M(t)
        #     trades.append(('Sell1', n1_sell, S1[-1],'Hold2', 0, S2[-1]))  # 记录交易
        # 
        # if act1 == 'Sell' and act2 == 'Sell': # 按照g平均分配N1和N2卖出
        #     n1_sell = round(g * N1[-1])  # 卖出的股票1数量, 取整
        #     n2_sell = round(g * N2[-1])  # 卖出的股票2数量, 取整
        #     if (N1[-1]-n1_sell)/(N1[-1]-n1_sell + N2[-1]-n2_sell) < p0:  # 调整N1/N2比例不低于p0
        #         n1_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0))
        #         n2_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)
        #         n1_sell = N1[-1] - n1_end
        #         n2_sell = N2[-1] - n2_end
        #     N1.append(N1[-1]-n1_sell)  # 更新N1(t)
        #     N2.append(N2[-1]-n2_sell)  # 更新N2(t)
        #     M.append(M[-1]+n1_sell*S1[-1]+n2_sell*S2[-1]) # 更新M(t)
        #     trades.append(('Sell1', n1_sell, S1[-1],'Sell2', n2_sell, S2[-1]))  # 记录交易


