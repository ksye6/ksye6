import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
# akshare as ak
# ��ȡ����۸�ak.fund_open_fund_info_em(symbo]="380005",indicator="��λ��ֵ����")
### 1 Data Preprocessing 

# ����
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


# ��ͨ
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

# ������㺯��
def calculate_all(t, h):
    X1_returns = pd.Series(X1_past[max(t-h,0):t], name='����')
    X2_returns = pd.Series(X2_past[max(t-h,0):t], name='��ͨ')
    X_returns = pd.concat([X1_returns,X2_returns],axis=1)
    mean_return = X_returns.mean()
    var_return = X_returns.var()
    cov_return =  X_returns.cov()
    p0 = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
    S0 = (p0 * S1_past[max(t-h,0):t]+(1-p0)*S2_past[max(t-h,0):t]).iloc[-1]
    # y1 = [np.mean(S0.pct_change().dropna())-0.02/360]/np.std(S0.pct_change().dropna())
    y0 = [p0 * np.mean(S1_past[max(t-h,0):t].pct_change().dropna())+(1-p0)*np.mean(S2_past[max(t-h,0):t].pct_change().dropna())-0.02/360] / (p0**2*var_return.iloc[0]+(1-p0)**2*var_return.iloc[1]+2*p0*(1-p0)*cov_return.iloc[0,1])**0.5
    return p0,S0,y0

# ������ƺ���
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

# ���� h ֵ
h_values = [30, 100, 300, len(X1_past)]

# ��������
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
logistic_cdf_values = 1 / (1 + np.exp(-X1_b * (xranges - X1_star))) # �������
# logistic_cdf_values = logistic.cdf(xranges, loc=np.mean(X1_past), scale=np.std(X1_past)) ���Դ��ĺ���ƫ��ϴ�

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

# ���������ۻ��ֲ����� CDF[X(t) = x | Y(t+1) = y]
def conditional_cdf(X, Y, x, y, assump = True):
    filtered_X = X[:-1][Y[1:] == y]  # ѡ�� Y(t+1) = y ��Ӧ�� X(t)
    if assump == 'Logistic':
        return logistic.cdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X)) # ���Դ���Logistic
    if assump == 'Normal':
        return norm.cdf(x, loc=np.mean(filtered_X), scale=np.std(filtered_X)) # ���Դ���Normal
    return np.sum(filtered_X <= x) / len(filtered_X) # CDF����

# ���� x ֵ�ķ�Χ
xranges = np.linspace(-0.05, 0.05, 100)

## ���� conditional CDF
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

## ���� conditional PDF,ѡ����̬����
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
    x = X[-1]  # �۲�ֵ X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors[y] * conditional_pdf(X1_past, Y_digi, x, y)  # ���� q(y) * f_y(x)
        if prob > max_prob:
            max_prob = prob
            y_star = y
    return y_star

y_star = compute_y_star(X1_past, Y_digi)

# ����ؼ�xֵ
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

# ��ͼ�ϱ�ǹؼ�ֵ
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

# �������ݼ�
tent = []
for i in range(len(Y_digi) - 6 + 1):
    row = Y_digi[i:i+6]
    tent.append(row)
# ����DataFrame
df = pd.DataFrame(tent, columns=[f'Y_{i}' for i in range(1, 7)])
print(df)

df_encoded = pd.get_dummies(df)
df = df_encoded.astype(int)

from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001)
# ɸѡ���Ϲ�����ʽ�Ĺ�������
rules = rules[rules['antecedents'].apply(lambda x: len(x) == 5) &rules['consequents'].apply(lambda x: len(x) == 1)]
filtered_rules = rules[~rules['antecedents'].apply(lambda x: 'Y_6_D' in x or 'Y_6_H' in x or 'Y_6_U' in x)]

# ��֧�ֶ�����
rules_support_top10 = filtered_rules.sort_values(by='support', ascending=False).head(10)
rules_support_top10['antecedents'] = rules_support_top10['antecedents'].apply(lambda x: sorted(x, key=lambda y: int(y.split('_')[1])))
rules_support_top10.loc[:, ['antecedents', 'consequents', 'support']]

# �����Ŷ�����
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

# �Ա�
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
# ������ɼ�̫�ÿ϶���������ֻ��ǰһ������ݣ���500��
# ǰһ��Ĺɼ�ͼ
plt.figure(figsize=(12, 12))
plt.plot(S1_future[:500])
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price of 0066.HK")
plt.show()

# ������Ԥ�⺯��
def compute_y_star(X, Y):
    x = X[-1]  # �۲�ֵ X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors[y] * conditional_pdf(X, Y, x, y)  # ���� q(y) * f_y(x)
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
    M = [100000]  # ��ʼ�ʽ�
    N = [0]  # ��ʼ��Ʊ����
    S = [S1_past.iloc[-1]]  # ��ʼ��Ʊ�۸�
    
    portfolio_values = [100000]  # Ͷ����ϼ�ֵ�б�
    government_values = [100000] # ��ծ��ֵ�仯�б�
    trades = []  # ���׼�¼�б�
    
    S_past = S1_past.copy()
    S_true = S1_future.copy()[:500]
    X_past = X1_past.copy()
    
    for t in range(0, len(S_true)):
        
        Y_past = digitize_X(X_past, epsilon)
        
        y_star = compute_y_star(X_past, Y_past)  # Ԥ����һ��ķ���
        today = digitize_X([X_past[-1]], epsilon) # ����ķ���
        _, _, macd = talib.MACD(S_past, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_new = macd.iloc[-1] # �����macdֵ
        delta = macd.iloc[-1]-macd.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        act = action(today,y_star,macd_new,delta) # �������������ǳ���
        
        # ����Ԥ������������
        if act == 'Buy':
            shares_to_buy = round(g * M[-1] / S[-1])  # ����Ĺ�Ʊ����, ȡ��
            N.append(N[-1]+shares_to_buy)  # ����N(t)
            M.append(M[-1]-shares_to_buy*S[-1]) # ����M(t)
            trades.append(('Buy', shares_to_buy, S[-1]))  # ��¼����
        if act == 'Sell':
            shares_to_sell = round(g * N[-1])  # ������Ʊ������, ȡ��
            N.append(N[-1]-shares_to_sell)  # ����N(t)
            M.append(M[-1]+shares_to_sell*S[-1]) # ����M(t)
            trades.append(('Sell', shares_to_sell, S[-1]))  # ��¼����
        if act == 'Hold':
            trades.append(('Hold', 0, S[-1]))  # ��¼����
        
        government_values.append(100000 * (1 + r) ** t)  # ���¹�ծ��ֵ
        S.append(S_true.iloc[t])  # ����ʵ�ʹɼ�S(t)
        portfolio_values.append(M[-1]+N[-1]*S[-1])  # ����Ͷ����ϼ�ֵ
        
        S_past = pd.concat([S_past, S_true.iloc[[t]]]) # ������ʷ�ɼ�����
        X_past = np.append(X_past, np.log(S_past.iloc[-1] / S_past.iloc[-2])) # ������ʷ�ɼ�������������

    return portfolio_values, government_values


g_aggressive = 0.5  # ������̰���Ȳ���
g_middle = 0.2
g_conservative = 0.05  # ���ص�̰���Ȳ���

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
    M = [100000]  # ��ʼ�ʽ�
    S1 = [S1_past.iloc[-1]]  # ��ʼ��Ʊ1�۸�
    S2 = [S2_past.iloc[-1]]  # ��ʼ��Ʊ2�۸�
    N1 = [round(100000*p0/S1[0])]  # ��ʼ��Ʊ1����
    N2 = [round(100000*(1-p0)/S2[0])]  # ��ʼ��Ʊ2����
    
    portfolio_values = [100000]  # Ͷ����ϼ�ֵ�б�
    government_values = [100000] # ��ծ��ֵ�仯�б�
    trades = []  # ���׼�¼�б�
    
    # ����Ԥ��S2
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    
    S2_true_ = S2_future.copy()[:500]
    
    for t in range(0, len(S1_true_)):
        
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y_star(X1_past_, Y1_past)  # Ԥ����һ��ķ���
        today1 = digitize_X([X1_past_[-1]], epsilon) # ����ķ���
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # �����macdֵ
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        act1 = action(today1,y1_star,macd1_new,delta1) # �������������ǳ���
        
        # ����Ԥ������������
        if act1 == 'Buy':
            shares_to_sell = round(g * N2[-1])  # ����N2������, ȡ��
            tent_money = shares_to_sell * S2[-1] # ��ת�۸�
            shares_to_buy = round(tent_money / S1[-1]) # ����N1������, ȡ��
            N1.append(N1[-1]+shares_to_buy)  # ����N1(t)
            N2.append(N2[-1]-shares_to_sell)  # ����N2(t)
            trades.append(('Buy', shares_to_buy, S1[-1]))  # ��¼S1�Ľ���
        if act1 == 'Sell':
            shares_to_sell = round(g * N1[-1])  # ����N1������, ȡ��
            tent_money = shares_to_sell * S1[-1] # ��ת�۸�
            shares_to_buy = round(tent_money / S2[-1]) # ����N2������, ȡ��
            N1.append(N1[-1]-shares_to_sell)  # ����N1(t)
            N2.append(N2[-1]+shares_to_buy)  # ����N2(t)
            trades.append(('Sell', shares_to_sell, S1[-1]))  # ��¼S1�Ľ���
        if act1 == 'Hold':
            trades.append(('Hold', 0, S1[-1]))  # ��¼����
        
        government_values.append(100000 * (1 + r) ** t)  # ���¹�ծ��ֵ
        S1.append(S1_true_.iloc[t])  # ����ʵ�ʹɼ�S1(t)
        S2.append(S2_true_.iloc[t])  # ����ʵ�ʹɼ�S2(t)
        portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # ����Ͷ����ϼ�ֵ
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # ������ʷ�ɼ�S1����
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # ������ʷ�ɼ���������X1����
    
    return portfolio_values, government_values

g_aggressive = 0.5  # ������̰���Ȳ���
# g_middle = 0.2
g_conservative = 0.05  # ���ص�̰���Ȳ���

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
# �����Ĳ��Ը��ã�����ԭ����S2һֱ�µ���̫���ؾͻ�һֱ����

# Ϊ�˷��������Ϊƽ�ȵĹ�Ʊ,���ǰ������ó�16.4��������,���ǿ���
S2_future = pd.Series(index=S2_future.index)
previous_value =16.4  # ��ʼֵ
np.random.seed(17)
for date in S2_future.index:
    random_value = np.random.uniform(previous_value - 0.2, previous_value + 0.2)
    random_value = max(12, min(20, random_value))  # ȷ�����ֵ�ڷ�Χ��
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

# �������ʱ����߲��

## 8.1 Efficient Frontier
# ��ԭS2_future
S2_future = S2[int(len(S2)/4*3)+1:]

def trade_strategy_2_1(g, r=0.001/100, p0 = 0.38):
    M = [100000]  # ��ʼ�ʽ�
    S1 = [S1_past.iloc[-1]]  # ��ʼ��Ʊ1�۸�
    S2 = [S2_past.iloc[-1]]  # ��ʼ��Ʊ2�۸�
    N1 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0))]  # ��ʼ��Ʊ1����
    N2 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)]  # ��ʼ��Ʊ2����
    P0t= [p0]
    Pt = [p0]
    
    portfolio_values = [100000]  # Ͷ����ϼ�ֵ�б�
    government_values = [100000] # ��ծ��ֵ�仯�б�
    trades = []  # ���׼�¼�б�
    
    # ����Ԥ��S2
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    
    S2_true_ = S2_future.copy()[:500]
    S2_past_ = S2_past.copy()
    X2_past_ = X2_past.copy()
    
    for t in range(0, len(S1_true_)):
        
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y_star(X1_past_, Y1_past)  # Ԥ����һ��ķ���
        today1 = digitize_X([X1_past_[-1]], epsilon) # ����ķ���
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # �����macdֵ
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        act1 = action(today1,y1_star,macd1_new,delta1) # �������������ǳ���
        
        # ����Ԥ������������
        if act1 == 'Buy':
            shares_to_sell = round(g * N2[-1])  # ����N2������, ȡ��
            tent_money = shares_to_sell * S2[-1] # ��ת�۸�
            shares_to_buy = round(tent_money / S1[-1]) # ����N1������, ȡ��
            N1.append(N1[-1]+shares_to_buy)  # ����N1(t)
            N2.append(N2[-1]-shares_to_sell)  # ����N2(t)
            trades.append(('Buy', shares_to_buy, S1[-1]))  # ��¼S1�Ľ���
        if act1 == 'Sell':
            shares_to_sell = round(g * N1[-1])  # ����N1������, ȡ��
            tent_money = shares_to_sell * S1[-1] # ��ת�۸�
            shares_to_buy = round(tent_money / S2[-1]) # ����N2������, ȡ��
            N1.append(N1[-1]-shares_to_sell)  # ����N1(t)
            N2.append(N2[-1]+shares_to_buy)  # ����N2(t)
            trades.append(('Sell', shares_to_sell, S1[-1]))  # ��¼S1�Ľ���
        if act1 == 'Hold':
            trades.append(('Hold', 0, S1[-1]))  # ��¼����
        
        government_values.append(100000 * (1 + r) ** t)  # ���¹�ծ��ֵ
        S1.append(S1_true_.iloc[t])  # ����ʵ�ʹɼ�S1(t)
        S2.append(S2_true_.iloc[t])  # ����ʵ�ʹɼ�S2(t)
        portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # ����Ͷ����ϼ�ֵ
        Pt.append(N1[-1]*S1[-1]/portfolio_values[-1]) # ����Pt
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # ������ʷ�ɼ�S1����
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # ������ʷ�ɼ���������X1����
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # ������ʷ�ɼ�S2����
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # ������ʷ�ɼ���������X2����
        
        X_returns = pd.concat([pd.Series(X1_past_, name='����'),pd.Series(X2_past_, name='��ͨ')],axis=1)
        mean_return = X_returns.mean()
        var_return = X_returns.var()
        cov_return =  X_returns.cov()
        p0new = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
        P0t.append(p0new)  # ����P0
    
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
    M = [100000]  # ��ʼ�ʽ�
    S1 = [S1_past.iloc[-1]]  # ��ʼ��Ʊ1�۸�
    S2 = [S2_past.iloc[-1]]  # ��ʼ��Ʊ2�۸�
    N1 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0))]  # ��ʼ��Ʊ1����
    N2 = [round(M[-1]/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)]  # ��ʼ��Ʊ2����
    P0t= [p0]
    Pt = [p0]
    result = []
    
    portfolio_values = [100000]  # Ͷ����ϼ�ֵ�б�
    government_values = [100000] # ��ծ��ֵ�仯�б�
    trades = []  # ���׼�¼�б�
    
    # ����Ԥ��S2
    S1_past_ = S1_past.copy()
    S1_true_ = S1_future.copy()[:500]
    X1_past_ = X1_past.copy()
    
    S2_true_ = S2_future.copy()[:500]
    S2_past_ = S2_past.copy()
    X2_past_ = X2_past.copy()

    X_returns = pd.concat([pd.Series(X1_past_, name='����'),pd.Series(X2_past_, name='��ͨ')],axis=1)
    mean_return = [X_returns.mean()]
    var_return = [X_returns.var()]
    cov_return =  [X_returns.cov()]
    sigma0 = p0**2*var_return[-1].iloc[0]+(1-p0)**2*var_return[-1].iloc[1]+2*p0*(1-p0)*cov_return[-1].iloc[0,1]
    sigma1 = X1_past_.var()
    sigmat = [sigma0]

    # �Ƚ���һ��
    Y1_past = digitize_X(X1_past_, epsilon)
    
    y1_star = compute_y_star(X1_past_, Y1_past)  # Ԥ����һ��ķ���
    today1 = digitize_X([X1_past_[-1]], epsilon) # ����ķ���
    _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
    macd1_new = macd1.iloc[-1] # �����macdֵ
    delta1 = macd1.iloc[-1]-macd1.iloc[-2] # ����Ա�����Ĳ�ֵ
    
    act1 = action(today1,y1_star,macd1_new,delta1) # �������������ǳ���
    
    # ����Ԥ������������
    if act1 == 'Buy':
        shares_to_sell = round(g * N2[-1])  # ����N2������, ȡ��
        tent_money = shares_to_sell * S2[-1] # ��ת�۸�
        shares_to_buy = round(tent_money / S1[-1]) # ����N1������, ȡ��
        N1.append(N1[-1]+shares_to_buy)  # ����N1(t)
        N2.append(N2[-1]-shares_to_sell)  # ����N2(t)
        trades.append(('Buy', shares_to_buy, S1[-1]))  # ��¼S1�Ľ���
    if act1 == 'Sell':
        shares_to_sell = round(g * N1[-1])  # ����N1������, ȡ��
        tent_money = shares_to_sell * S1[-1] # ��ת�۸�
        shares_to_buy = round(tent_money / S2[-1]) # ����N2������, ȡ��
        N1.append(N1[-1]-shares_to_sell)  # ����N1(t)
        N2.append(N2[-1]+shares_to_buy)  # ����N2(t)
        trades.append(('Sell', shares_to_sell, S1[-1]))  # ��¼S1�Ľ���
    if act1 == 'Hold':
        trades.append(('Hold', 0, S1[-1]))  # ��¼����
    
    government_values.append(100000 * (1 + r) ** 0)  # ���¹�ծ��ֵ
    S1.append(S1_true_.iloc[0])  # ����ʵ�ʹɼ�S1(t)
    S2.append(S2_true_.iloc[0])  # ����ʵ�ʹɼ�S2(t)
    portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # ����Ͷ����ϼ�ֵ
    Pt.append(N1[-1]*S1[-1]/portfolio_values[-1]) # ����Pt
    
    S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[0]]]) # ������ʷ�ɼ�S1����
    X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # ������ʷ�ɼ���������X1����
    S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[0]]]) # ������ʷ�ɼ�S2����
    X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # ������ʷ�ɼ���������X2����
    
    X_returns = pd.concat([pd.Series(X1_past_, name='����'),pd.Series(X2_past_, name='��ͨ')],axis=1)
    mean_return.append(X_returns.mean())
    var_return.append(X_returns.var())
    cov_return.append(X_returns.cov())
    p0new = (var_return[-1].iloc[1]-cov_return[-1].iloc[0,1])/(var_return[-1].iloc[0]+var_return[-1].iloc[1]-2*cov_return[-1].iloc[0,1])
    P0t.append(p0new)  # ����P0
    
    sigmat.append(Pt[-1]**2*var_return[-1].iloc[0]+(1-Pt[-1])**2*var_return[-1].iloc[1]+2*Pt[-1]*(1-Pt[-1])*cov_return[-1].iloc[0,1]) # ����sigmat

    # ��������sigmatѡ��ni
    for t in range(1, len(S1_true_)):
        Y1_past = digitize_X(X1_past_, epsilon)
        
        y1_star = compute_y_star(X1_past_, Y1_past)  # Ԥ����һ��ķ���
        today1 = digitize_X([X1_past_[-1]], epsilon) # ����ķ���
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # �����macdֵ
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        act1 = action(today1,y1_star,macd1_new,delta1) # �������������ǳ���
        
        if act1 == 'Buy':
            sig = sigmat[-1]+g*(sigma1-sigmat[-1])
        if act1 == 'Sell':
            sig = sigmat[-1]-g*(sigmat[-1]-sigma0)
        
        a = sp.Symbol('a')
        # ���巽��
        equation = sp.Eq(sig, a**2 * var_return[-1].iloc[0] + (1 - a)**2 * var_return[-1].iloc[1] + 2 * a * (1 - a) * cov_return[-1].iloc[0,1])
        # �ⷽ��
        solutions = sp.solve(equation, a)
        filter_ = [x for x in solutions if (not isinstance(x, sp.core.mul.Mul) and not isinstance(x, sp.core.add.Add))]
        filter_ = [x for x in filter_ if P0t[-1] <= x <= 1]

        if len(filter_)!=0:
            result.append(filter_)
        
        finalN1 = round(result[-1][-1]*portfolio_values[-1]/S1[-1])
        finalN2 = round((1-result[-1][-1])*portfolio_values[-1]/S2[-1])

        # ����Ԥ������������
        if N1[-1] < finalN1:
            shares_to_buy = finalN1 - N1[-1] # ����N1������, ȡ��
            N1.append(finalN1)  # ����N1(t)
            N2.append(finalN2)  # ����N2(t)
            trades.append(('Buy', shares_to_buy, S1[-1]))  # ��¼S1�Ľ���
        if N1[-1] > finalN1:
            shares_to_sell = N1[-1] - finalN1  # ����N1������, ȡ��
            N1.append(finalN1)  # ����N1(t)
            N2.append(finalN2)  # ����N2(t)
            trades.append(('Sell', shares_to_sell, S1[-1]))  # ��¼S1�Ľ���
        if act1 == 'Hold':
            trades.append(('Hold', 0, S1[-1]))  # ��¼����

        government_values.append(100000 * (1 + r) ** t)  # ���¹�ծ��ֵ
        S1.append(S1_true_.iloc[t])  # ����ʵ�ʹɼ�S1(t)
        S2.append(S2_true_.iloc[t])  # ����ʵ�ʹɼ�S2(t)
        portfolio_values.append(N1[-1]*S1[-1]+N2[-1]*S2[-1])  # ����Ͷ����ϼ�ֵ
        Pt.append(result[-1][-1]) # ����Pt

        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # ������ʷ�ɼ�S1����
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # ������ʷ�ɼ���������X1����
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # ������ʷ�ɼ�S2����
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # ������ʷ�ɼ���������X2����

        X_returns = pd.concat([pd.Series(X1_past_, name='����'),pd.Series(X2_past_, name='��ͨ')],axis=1)
        mean_return.append(X_returns.mean())
        var_return.append(X_returns.var())
        cov_return.append(X_returns.cov())
        p0new = (var_return[-1].iloc[1]-cov_return[-1].iloc[0,1])/(var_return[-1].iloc[0]+var_return[-1].iloc[1]-2*cov_return[-1].iloc[0,1])
        P0t.append(p0new)  # ����P0

        sigmat.append(Pt[-1]**2*var_return[-1].iloc[0]+(1-Pt[-1])**2*var_return[-1].iloc[1]+2*Pt[-1]*(1-Pt[-1])*cov_return[-1].iloc[0,1]) # ����sigmat

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

# ȷʵ�Ƚ��˺ܶ�


### 9 A Portfolio with Two Stocks and Money
epsilon = 0.004

# X to Y
Y1_digi = digitize_X(X1_past, epsilon)
Y2_digi = digitize_X(X2_past, epsilon)

## ���� conditional PDF,ѡ����̬����
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
    x = X[-1]  # �۲�ֵ X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors1[y] * conditional_pdf(X, Y, x, y)  # ���� q(y) * f_y(x)
        if prob > max_prob:
            max_prob = prob
            y_star = y
    return y_star

def compute_y2_star(X, Y):
    x = X[-1]  # �۲�ֵ X(t)
    max_prob = float('-inf')
    y_star = None
    for y in ['D', 'U', 'H']:
        prob = priors2[y] * conditional_pdf(X, Y, x, y)  # ���� q(y) * f_y(x)
        if prob > max_prob:
            max_prob = prob
            y_star = y
    return y_star

# y1_star = compute_y1_star(X1_past, Y1_digi)
# y2_star = compute_y2_star(X2_past, Y2_digi)


def trade_strategy_3(g, r=0.001/100,p0 = 0.38):
    M = [100000]  # ��ʼ�ʽ�
    N1 = [0]  # ��ʼ��Ʊ1����
    N2 = [0]  # ��ʼ��Ʊ2����
    S1 = [S1_past.iloc[-1]]  # ��ʼ��Ʊ�۸�
    S2 = [S2_past.iloc[-1]]  # ��ʼ��Ʊ�۸�
    
    portfolio_values = [100000]  # Ͷ����ϼ�ֵ�б�
    government_values = [100000] # ��ծ��ֵ�仯�б�
    trades = []  # ���׼�¼�б�
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
        
        y1_star = compute_y1_star(X1_past_, Y1_past)  # Ԥ����һ��ķ���
        today1 = digitize_X([X1_past_[-1]], epsilon) # ����ķ���
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # �����macdֵ
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        act1 = action(today1,y1_star,macd1_new,delta1) # ������Ʊ1���������ǳ���
        
        Y2_past = digitize_X(X2_past_, epsilon)
        
        y2_star = compute_y2_star(X2_past_, Y2_past)  # Ԥ����һ��ķ���
        today2 = digitize_X([X2_past_[-1]], epsilon) # ����ķ���
        _, _, macd2 = talib.MACD(S2_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd2_new = macd2.iloc[-1] # �����macdֵ
        delta2 = macd2.iloc[-1]-macd2.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        act2 = action(today2,y2_star,macd2_new,delta2) # ������Ʊ2���������ǳ���        
        
        # ����Ԥ������������
        if act1 == 'Buy' and act2 == 'Buy': # ����gƽ������M����1��2
            n1_buy = round(g * M[-1]/2 / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
            n2_buy = round(g * M[-1]/2 / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            if (N1[-1]+n1_buy)*S1[-1]/((N1[-1]+n1_buy)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*p0/S1[-1])
                n2_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*(1-p0)/S2[-1])
                n1_buy = n1_end - N1[-1]
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]+n1_buy)  # ����N1(t)
            N2.append(N2[-1]+n2_buy)  # ����N2(t)
            M.append(M[-1]-n1_buy*S1[-1]-n2_buy*S2[-1]) # ����M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����
        
        if act1 == 'Buy' and act2 == 'Hold': # ����g����M����1
            n1_buy = round(g * M[-1] / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
            N1.append(N1[-1]+n1_buy)  # ����N1(t)
            N2.append(N2[-1])  # ����N2(t)
            M.append(M[-1]-n1_buy*S1[-1]) # ����M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        
        if act1 == 'Buy' and act2 == 'Sell': # ����2,���ϰ���g�����M����1
            n2_sell = round(g * N2[-1])  # �����Ĺ�Ʊ2����, ȡ��
            n1_buy = round(g * (M[-1] + n2_sell*S2[-1]) / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
            N1.append(N1[-1]+n1_buy)  # ����N1(t)
            N2.append(N2[-1]-n2_sell)  # ����N2(t)
            M.append(M[-1]-n1_buy*S1[-1]+n2_sell*S2[-1]) # ����M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����        
        
        if act1 == 'Hold' and act2 == 'Buy': # ����g����M����2
            n2_buy = round(g * M[-1] / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            if N1[-1]*S1[-1]/(N1[-1]*S1[-1]+(N2[-1]+n2_buy)*S2[-1]) < p0:  # ����pt����������p0
                n2_end = round(N1[-1]*S1[-1]*(1-p0)/p0/S2[-1])
                n2_buy = max(n2_end - N2[-1],0)
            N1.append(N1[-1])  # ����N1(t)
            N2.append(N2[-1]+n2_buy)  # ����N2(t)
            M.append(M[-1]-n2_buy*S2[-1]) # ����M(t)
            trades.append(('Hold1', 0, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����        
        
        if act1 == 'Hold' and act2 == 'Hold': # ����
            N1.append(N1[-1])  # ����N1(t)
            N2.append(N2[-1])  # ����N2(t)
            M.append(M[-1]) # ����M(t)
            trades.append(('Hold1', 0, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        
        if act1 == 'Hold' and act2 == 'Sell': # ����g����N2����2
            n2_sell = round(g * N2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            N1.append(N1[-1])  # ����N1(t)
            N2.append(N2[-1]-n2_sell)  # ����N2(t)
            M.append(M[-1]+n2_sell*S2[-1]) # ����M(t)
            trades.append(('Hold1', 0, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����  
        
        if act1 == 'Sell' and act2 == 'Buy': # ����2,���ϰ���g�����M����1
            n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
            n2_buy = round(g * (M[-1] + n1_sell*S1[-1]) / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*p0/S1[-1])
                n2_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]-n1_sell)  # ����N1(t)
            N2.append(N2[-1]+n2_buy)  # ����N2(t)
            M.append(M[-1]+n1_sell*S1[-1]-n2_buy*S2[-1]) # ����M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����
        
        if act1 == 'Sell' and act2 == 'Hold': # ����g����N1����1
            n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1]+N2[-1]*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round(N2[-1]*S2[-1]*p0/(1-p0)/S1[-1])
                n1_sell = max(N1[-1]-n1_end,0)
            N1.append(N1[-1]-n1_sell)  # ����N1(t)
            N2.append(N2[-1])  # ����N2(t)
            M.append(M[-1]+n1_sell*S1[-1]) # ����M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        
        if act1 == 'Sell' and act2 == 'Sell': # ����gƽ������N1��N2����
            n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
            n2_sell = round(g * N2[-1])  # �����Ĺ�Ʊ2����, ȡ��
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]-n2_sell)*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*p0/S1[-1])
                n2_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_sell = N2[-1] - n2_end
            N1.append(N1[-1]-n1_sell)  # ����N1(t)
            N2.append(N2[-1]-n2_sell)  # ����N2(t)
            M.append(M[-1]+n1_sell*S1[-1]+n2_sell*S2[-1]) # ����M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����
        
        government_values.append(100000 * (1 + r) ** t)  # ���¹�ծ��ֵ
        S1.append(S1_true_.iloc[t])  # ����ʵ�ʹɼ�S1(t)
        S2.append(S2_true_.iloc[t])  # ����ʵ�ʹɼ�S2(t)
        portfolio_values.append(M[-1]+N1[-1]*S1[-1]+N2[-1]*S2[-1])  # ����Ͷ����ϼ�ֵ
        Pt.append(N1[-1]*S1[-1]/(N1[-1]*S1[-1]+N2[-1]*S2[-1])) # ����Pt
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # ������ʷ�ɼ�S1����
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # ������ʷ�ɼ���������X1����
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # ������ʷ�ɼ�S2����
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # ������ʷ�ɼ���������X2����
        X_returns = pd.concat([pd.Series(X1_past_, name='����'),pd.Series(X2_past_, name='��ͨ')],axis=1)
        
        mean_return = X_returns.mean()
        var_return = X_returns.var()
        cov_return =  X_returns.cov()
        p0new = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
        P0t.append(p0new)  # ����P0

    return portfolio_values, government_values


g_aggressive = 0.5  # ������̰���Ȳ���
g_middle = 0.2
g_conservative = 0.05  # ���ص�̰���Ȳ���

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

# �����д��


### ����RSIָ�� ���ǿ��ָ�� �������� �������� �ȿ��Ƕ������ʵ��
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

def trade_strategy_4(g,a=0.65, r=0.001/100):#�������õ�a�Ƿ������ֻ��Ʊ�ʽ�ı�������������a=0.5��������ֻ��Ʊ��ද�õ��ʽ����5��
    M = [100000]  # ��ʼ�ʽ�
    N_1 = [0]  # S1��ʼ��Ʊ����
    N_2 = [0]  #S2��ʼ��Ʊ����    
    S_1 = [S1_past.iloc[-1]]  # S1��ʼ��Ʊ�۸�
    S_2= [S2_past.iloc[-1]] #S2��ʼ��Ʊ�۸�
    M_1=[elem * a for elem in M]
    M_2=[elem * (1-a) for elem in M]
    portfolio_values = [100000]  # Ͷ����ϼ�ֵ�б�
    government_values = [100000] # ��ծ��ֵ�仯�б�
    trades_S1 = [] # S1���׼�¼�б�
    trades_S2 = [] # S2���׼�¼�б�
    
    S1_pas = S1_past.copy()
    S2_pas=S2_past.copy()
    S1_true = S1_future.copy()[:500]
    S2_true=S2_future.copy()[:500]
    X1_pas = X1_past.copy()
    X2_pas=X2_past.copy()
    
    for t in range(0, len(S1_true)):
        
        Y1_past = digitize_X(X1_pas, epsilon)
        Y2_past=digitize_X(X2_pas,epsilon)
        y_star_S1 = compute_y1_star(X1_pas, Y1_past)  # ��ƱS1Ԥ����һ��ķ���
        y_star_S2 = compute_y2_star(X2_pas, Y2_past)  # ��ƱS2Ԥ����һ��ķ���
        today_S1 = digitize_X([X1_pas[-1]], epsilon) # ��ƱS1����ķ���
        today_S2 = digitize_X([X2_pas[-1]], epsilon)#��ƱS2����ķ���
        _, _, macd_S1 = talib.MACD(S1_pas, fastperiod=12, slowperiod=26, signalperiod=9)
        _, _, macd_S2 = talib.MACD(S2_pas, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_new_S1 = macd_S1.iloc[-1] # �����macdֵ
        macd_new_S2 =macd_S2.iloc[-1]
        delta_S1 = macd_S1.iloc[-1]-macd_S1.iloc[-2] # ����Ա�����Ĳ�ֵ
        delta_S2 = macd_S2.iloc[-1]-macd_S2.iloc[-2]
        rsi_S1=rsi_indicator(S1_pas[-15:], window=14).iloc[-1] #�����ڽ����������ƱS1��RSIֵ�����ֵ>70,������<30����30-70֮��Ͳ���
        rsi_S2=rsi_indicator(S2_pas[-15:], window=14).iloc[-1] #�����ڽ����������ƱS2��RSIֵ�����ֵ>70,������<30����30-70֮��Ͳ���
        act_S1 = action2(today_S1,y_star_S1,macd_new_S1,delta_S1,rsi_S1) # ���ڹ�ƱS1�������������ǳ���
        act_S2=action2(today_S2,y_star_S2,macd_new_S2,delta_S2,rsi_S2) # ���ڹ�ƱS2�������������ǳ���
        
        # ����Ԥ������������
        if act_S1 == 'Buy':
            shares_to_buy = round(g * M_1[-1] / S_1[-1])  # ����Ĺ�Ʊ����, ȡ��
            N_1.append(N_1[-1]+shares_to_buy)  # ����N(t)
            M_1.append(M_1[-1]-shares_to_buy*S_1[-1]) # ����M(t)
            trades_S1.append(('Buy', shares_to_buy, S_1[-1]))  # ��¼����
        
        if act_S1 == 'Sell':
            shares_to_sell = round(g * N_1[-1])  # ������Ʊ������, ȡ��
            N_1.append(N_1[-1]-shares_to_sell)  # ����N(t)
            M_1.append(M_1[-1]+shares_to_sell*S_1[-1]) # ����M(t)
            trades_S1.append(('Sell', shares_to_sell, S_1[-1]))  # ��¼����
        if act_S1 == 'Hold':
            trades_S1.append(('Hold', 0, S_1[-1]))  # ��¼����

        if act_S2 == 'Buy':
            shares_to_buy = round(g * M_2[-1] / S_2[-1])  # ����Ĺ�Ʊ����, ȡ��
            N_2.append(N_2[-1]+shares_to_buy)  # ����N(t)
            M_2.append(M_2[-1]-shares_to_buy*S_2[-1]) # ����M(t)
            trades_S2.append(('Buy', shares_to_buy, S_2[-1]))  # ��¼����
        
        if act_S2 == 'Sell':
            shares_to_sell = round(g * N_2[-1])  # ������Ʊ������, ȡ��
            N_2.append(N_2[-1]-shares_to_sell)  # ����N(t)
            M_2.append(M_2[-1]+shares_to_sell*S_2[-1]) # ����M(t)
            trades_S2.append(('Sell', shares_to_sell, S_2[-1]))  # ��¼����
        if act_S2 == 'Hold':
            trades_S2.append(('Hold', 0, S_2[-1]))  # ��¼����
        
        government_values.append(100000 * (1 + r) ** t)  # ���¹�ծ��ֵ
        S_1.append(S1_true.iloc[t])  # ����ʵ�ʹɼ�S1(t)
        S_2.append(S2_true.iloc[t])  # ����ʵ�ʹɼ�S1(t)
        portfolio_values.append(M_1[-1]+N_1[-1]*S_1[-1]+M_2[-1]+N_2[-1]*S_2[-1])  # ����Ͷ����ϼ�ֵ
        
        S1_pas = pd.concat([S1_pas, S1_true.iloc[[t]]]) # ����S1��ʷ�ɼ�����
        S2_pas= pd.concat([S2_pas, S2_true.iloc[[t]]]) #����S2��ʷ�ɼ�����
        X1_pas = np.append(X1_pas, np.log(S1_pas.iloc[-1] / S1_pas.iloc[-2])) # ������ʷ�ɼ�������������
        X2_pas = np.append(X2_pas, np.log(S2_pas.iloc[-1] / S2_pas.iloc[-2]))

    return portfolio_values, government_values
g_aggressive = 0.5  # ������̰���Ȳ���
g_middle = 0.2
g_conservative = 0.05 # ���ص�̰���Ȳ���
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


### �ۺ�section7, section8, RSI index
def trade_strategy_5(g, r=0.001/100,p0 = 0.38 ,a=0.65):
    M = [100000]  # ��ʼ�ʽ�
    N1 = [0]  # ��ʼ��Ʊ1����
    N2 = [0]  # ��ʼ��Ʊ2����
    S1 = [S1_past.iloc[-1]]  # ��ʼ��Ʊ�۸�
    S2 = [S2_past.iloc[-1]]  # ��ʼ��Ʊ�۸�
    
    portfolio_values = [100000]  # Ͷ����ϼ�ֵ�б�
    government_values = [100000] # ��ծ��ֵ�仯�б�
    trades = []  # ���׼�¼�б�
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
        
        y1_star = compute_y1_star(X1_past_, Y1_past)  # Ԥ����һ��ķ���
        today1 = digitize_X([X1_past_[-1]], epsilon) # ����ķ���
        _, _, macd1 = talib.MACD(S1_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd1_new = macd1.iloc[-1] # �����macdֵ
        delta1 = macd1.iloc[-1]-macd1.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        Y2_past = digitize_X(X2_past_, epsilon)
        
        y2_star = compute_y2_star(X2_past_, Y2_past)  # Ԥ����һ��ķ���
        today2 = digitize_X([X2_past_[-1]], epsilon) # ����ķ���
        _, _, macd2 = talib.MACD(S2_past_, fastperiod=12, slowperiod=26, signalperiod=9)
        macd2_new = macd2.iloc[-1] # �����macdֵ
        delta2 = macd2.iloc[-1]-macd2.iloc[-2] # ����Ա�����Ĳ�ֵ
        
        rsi_S1=rsi_indicator(S1_past_[-15:], window=14).iloc[-1] #�����ڽ����������ƱS1��RSIֵ�����ֵ>70,������<30����30-70֮��Ͳ���
        rsi_S2=rsi_indicator(S2_past_[-15:], window=14).iloc[-1] #�����ڽ����������ƱS2��RSIֵ�����ֵ>70,������<30����30-70֮��Ͳ���
        
        
        act1 = action2(today1,y1_star,macd1_new,delta1,rsi_S1) # ���ڹ�ƱS1�������������ǳ���
        act2 = action2(today2,y2_star,macd2_new,delta2,rsi_S2) # ���ڹ�ƱS2�������������ǳ���
        
        # act1 = action(today1,y1_star,macd1_new,delta1) # ������Ʊ1���������ǳ���
        # act2 = action(today2,y2_star,macd2_new,delta2) # ������Ʊ2���������ǳ���   
        
        # ����Ԥ������������
        if act1 == 'Buy' and act2 == 'Buy': # ����gƽ������M����1��2
            n1_buy = round(g * M[-1]/2 / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
            n2_buy = round(g * M[-1]/2 / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            if (N1[-1]+n1_buy)*S1[-1]/((N1[-1]+n1_buy)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*p0/S1[-1])
                n2_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])*(1-p0)/S2[-1])
                n1_buy = n1_end - N1[-1]
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]+n1_buy)  # ����N1(t)
            N2.append(N2[-1]+n2_buy)  # ����N2(t)
            M.append(M[-1]-n1_buy*S1[-1]-n2_buy*S2[-1]) # ����M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����
        
        if act1 == 'Buy' and act2 == 'Hold': # ����g����M����1
            n1_buy = round(g * M[-1] / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
            N1.append(N1[-1]+n1_buy)  # ����N1(t)
            N2.append(N2[-1])  # ����N2(t)
            M.append(M[-1]-n1_buy*S1[-1]) # ����M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        
        if act1 == 'Buy' and act2 == 'Sell': # ����2,���ϰ���g�����M����1
            n2_sell = round(g * N2[-1])  # �����Ĺ�Ʊ2����, ȡ��
            n1_buy = round(g * (M[-1] + n2_sell*S2[-1]) / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
            N1.append(N1[-1]+n1_buy)  # ����N1(t)
            N2.append(N2[-1]-n2_sell)  # ����N2(t)
            M.append(M[-1]-n1_buy*S1[-1]+n2_sell*S2[-1]) # ����M(t)
            trades.append(('Buy1', n1_buy, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����        
        
        if act1 == 'Hold' and act2 == 'Buy': # ����g����M����2
            n2_buy = round(g * M[-1] / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            if N1[-1]*S1[-1]/(N1[-1]*S1[-1]+(N2[-1]+n2_buy)*S2[-1]) < p0:  # ����pt����������p0
                n2_end = round(N1[-1]*S1[-1]*(1-p0)/p0/S2[-1])
                n2_buy = max(n2_end - N2[-1],0)
            N1.append(N1[-1])  # ����N1(t)
            N2.append(N2[-1]+n2_buy)  # ����N2(t)
            M.append(M[-1]-n2_buy*S2[-1]) # ����M(t)
            trades.append(('Hold1', 0, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����        
        
        if act1 == 'Hold' and act2 == 'Hold': # ����
            N1.append(N1[-1])  # ����N1(t)
            N2.append(N2[-1])  # ����N2(t)
            M.append(M[-1]) # ����M(t)
            trades.append(('Hold1', 0, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        
        if act1 == 'Hold' and act2 == 'Sell': # ����g����N2����2
            n2_sell = round(g * N2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            N1.append(N1[-1])  # ����N1(t)
            N2.append(N2[-1]-n2_sell)  # ����N2(t)
            M.append(M[-1]+n2_sell*S2[-1]) # ����M(t)
            trades.append(('Hold1', 0, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����  
        
        if act1 == 'Sell' and act2 == 'Buy': # ����2,���ϰ���g�����M����1
            n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
            n2_buy = round(g * (M[-1] + n1_sell*S1[-1]) / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]+n2_buy)*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*p0/S1[-1])
                n2_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_buy = n2_end - N2[-1]
            N1.append(N1[-1]-n1_sell)  # ����N1(t)
            N2.append(N2[-1]+n2_buy)  # ����N2(t)
            M.append(M[-1]+n1_sell*S1[-1]-n2_buy*S2[-1]) # ����M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����
        
        if act1 == 'Sell' and act2 == 'Hold': # ����g����N1����1
            n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1]+N2[-1]*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round(N2[-1]*S2[-1]*p0/(1-p0)/S1[-1])
                n1_sell = max(N1[-1]-n1_end,0)
            N1.append(N1[-1]-n1_sell)  # ����N1(t)
            N2.append(N2[-1])  # ����N2(t)
            M.append(M[-1]+n1_sell*S1[-1]) # ����M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        
        if act1 == 'Sell' and act2 == 'Sell': # ����gƽ������N1��N2����
            n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
            n2_sell = round(g * N2[-1])  # �����Ĺ�Ʊ2����, ȡ��
            if (N1[-1]-n1_sell)*S1[-1]/((N1[-1]-n1_sell)*S1[-1] + (N2[-1]-n2_sell)*S2[-1]) < p0:  # ����pt����������p0
                n1_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*p0/S1[-1])
                n2_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])*(1-p0)/S2[-1])
                n1_sell = N1[-1] - n1_end
                n2_sell = N2[-1] - n2_end
            N1.append(N1[-1]-n1_sell)  # ����N1(t)
            N2.append(N2[-1]-n2_sell)  # ����N2(t)
            M.append(M[-1]+n1_sell*S1[-1]+n2_sell*S2[-1]) # ����M(t)
            trades.append(('Sell1', n1_sell, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����
        
        government_values.append(100000 * (1 + r) ** t)  # ���¹�ծ��ֵ
        S1.append(S1_true_.iloc[t])  # ����ʵ�ʹɼ�S1(t)
        S2.append(S2_true_.iloc[t])  # ����ʵ�ʹɼ�S2(t)
        portfolio_values.append(M[-1]+N1[-1]*S1[-1]+N2[-1]*S2[-1])  # ����Ͷ����ϼ�ֵ
        # Pt.append(N1[-1]*S1[-1]/(N1[-1]*S1[-1]+N2[-1]*S2[-1])) # ����Pt
        
        S1_past_ = pd.concat([S1_past_, S1_true_.iloc[[t]]]) # ������ʷ�ɼ�S1����
        X1_past_ = np.append(X1_past_, np.log(S1_past_.iloc[-1] / S1_past_.iloc[-2])) # ������ʷ�ɼ���������X1����
        S2_past_ = pd.concat([S2_past_, S2_true_.iloc[[t]]]) # ������ʷ�ɼ�S2����
        X2_past_ = np.append(X2_past_, np.log(S2_past_.iloc[-1] / S2_past_.iloc[-2])) # ������ʷ�ɼ���������X2����
        X_returns = pd.concat([pd.Series(X1_past_, name='����'),pd.Series(X2_past_, name='��ͨ')],axis=1)
        
        mean_return = X_returns.mean()
        var_return = X_returns.var()
        cov_return =  X_returns.cov()
        p0new = (var_return.iloc[1]-cov_return.iloc[0,1])/(var_return.iloc[0]+var_return.iloc[1]-2*cov_return.iloc[0,1])
        P0t.append(p0new)  # ����P0

    return portfolio_values, government_values


g_aggressive = 0.5  # ������̰���Ȳ���
g_middle = 0.2
g_conservative = 0.05  # ���ص�̰���Ȳ���

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

### 2024.5.5 ����: ������Ǵ�����룬�������N1/N2

        # # ����Ԥ������������
        # if act1 == 'Buy' and act2 == 'Buy': # ����gƽ������M����1��2
        #     n1_buy = round(g * M[-1]/2 / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
        #     n2_buy = round(g * M[-1]/2 / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
        #     if (N1[-1]+n1_buy)/(N1[-1]+n1_buy + N2[-1]+n2_buy) < p0:  # ����N1/N2����������p0
        #         n1_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0))
        #         n2_end = round((g*M[-1] + N1[-1]*S1[-1] + N2[-1]*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)
        #         n1_buy = n1_end - N1[-1]
        #         n2_buy = n2_end - N2[-1]
        #     N1.append(N1[-1]+n1_buy)  # ����N1(t)
        #     N2.append(N2[-1]+n2_buy)  # ����N2(t)
        #     M.append(M[-1]-n1_buy*S1[-1]-n2_buy*S2[-1]) # ����M(t)
        #     trades.append(('Buy1', n1_buy, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����
        # 
        # if act1 == 'Buy' and act2 == 'Hold': # ����g����M����1
        #     n1_buy = round(g * M[-1] / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
        #     N1.append(N1[-1]+n1_buy)  # ����N1(t)
        #     N2.append(N2[-1])  # ����N2(t)
        #     M.append(M[-1]-n1_buy*S1[-1]) # ����M(t)
        #     trades.append(('Buy1', n1_buy, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        # 
        # if act1 == 'Buy' and act2 == 'Sell': # ����2,���ϰ���g�����M����1
        #     n2_sell = round(g * N2[-1])  # �����Ĺ�Ʊ2����, ȡ��
        #     n1_buy = round(g * (M[-1] + n2_sell*S2[-1]) / S1[-1])  # ����Ĺ�Ʊ1����, ȡ��
        #     N1.append(N1[-1]+n1_buy)  # ����N1(t)
        #     N2.append(N2[-1]-n2_sell)  # ����N2(t)
        #     M.append(M[-1]-n1_buy*S1[-1]+n2_sell*S2[-1]) # ����M(t)
        #     trades.append(('Buy1', n1_buy, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����        
        # 
        # if act1 == 'Hold' and act2 == 'Buy': # ����g����M����2
        #     n2_buy = round(g * M[-1] / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
        #     if N1[-1]/(N1[-1]+N2[-1]+n2_buy) < p0:  # ����N1/N2����������p0
        #         n2_end = round(N1[-1]*(1-p0)/p0)
        #         n2_buy = max(n2_end - N2[-1],0)
        #     N1.append(N1[-1])  # ����N1(t)
        #     N2.append(N2[-1]+n2_buy)  # ����N2(t)
        #     M.append(M[-1]-n2_buy*S2[-1]) # ����M(t)
        #     trades.append(('Hold1', 0, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����        
        # 
        # if act1 == 'Hold' and act2 == 'Hold': # ����
        #     N1.append(N1[-1])  # ����N1(t)
        #     N2.append(N2[-1])  # ����N2(t)
        #     M.append(M[-1]) # ����M(t)
        #     trades.append(('Hold1', 0, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        # 
        # if act1 == 'Hold' and act2 == 'Sell': # ����g����N2����2
        #     n2_sell = round(g * N2[-1])  # ����Ĺ�Ʊ2����, ȡ��
        #     N1.append(N1[-1])  # ����N1(t)
        #     N2.append(N2[-1]-n2_sell)  # ����N2(t)
        #     M.append(M[-1]+n2_sell*S2[-1]) # ����M(t)
        #     trades.append(('Hold1', 0, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����  
        # 
        # if act1 == 'Sell' and act2 == 'Buy': # ����2,���ϰ���g�����M����1
        #     n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
        #     n2_buy = round(g * (M[-1] + n1_sell*S1[-1]) / S2[-1])  # ����Ĺ�Ʊ2����, ȡ��
        #     if (N1[-1]-n1_sell)/(N1[-1]-n1_sell + N2[-1]+n2_buy) < p0:  # ����N1/N2����������p0
        #         n1_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))/(S1[-1]+S2[-1]*(1-p0)/p0))
        #         n2_end = round(((N1[-1]-n1_sell)*S1[-1] + N2[-1]*S2[-1] + g * (M[-1] + n1_sell*S1[-1]))/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)
        #         n1_sell = N1[-1] - n1_end
        #         n2_buy = n2_end - N2[-1]
        #     N1.append(N1[-1]-n1_sell)  # ����N1(t)
        #     N2.append(N2[-1]+n2_buy)  # ����N2(t)
        #     M.append(M[-1]+n1_sell*S1[-1]-n2_buy*S2[-1]) # ����M(t)
        #     trades.append(('Sell1', n1_sell, S1[-1],'Buy2',n2_buy, S2[-1]))  # ��¼����
        # 
        # if act1 == 'Sell' and act2 == 'Hold': # ����g����N1����1
        #     n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
        #     if (N1[-1]-n1_sell)/(N1[-1]+N2[-1]-n1_sell) < p0:  # ����N1/N2����������p0
        #         n1_end = round(N2[-1]*p0/(1-p0))
        #         n1_sell = max(N1[-1]-n1_end,0)
        #     N1.append(N1[-1]-n1_sell)  # ����N1(t)
        #     N2.append(N2[-1])  # ����N2(t)
        #     M.append(M[-1]+n1_sell*S1[-1]) # ����M(t)
        #     trades.append(('Sell1', n1_sell, S1[-1],'Hold2', 0, S2[-1]))  # ��¼����
        # 
        # if act1 == 'Sell' and act2 == 'Sell': # ����gƽ������N1��N2����
        #     n1_sell = round(g * N1[-1])  # �����Ĺ�Ʊ1����, ȡ��
        #     n2_sell = round(g * N2[-1])  # �����Ĺ�Ʊ2����, ȡ��
        #     if (N1[-1]-n1_sell)/(N1[-1]-n1_sell + N2[-1]-n2_sell) < p0:  # ����N1/N2����������p0
        #         n1_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0))
        #         n2_end = round((N1[-1]*S1[-1] + N2[-1]*S2[-1]-n1_sell*S1[-1]-n2_sell*S2[-1])/(S1[-1]+S2[-1]*(1-p0)/p0)*(1-p0)/p0)
        #         n1_sell = N1[-1] - n1_end
        #         n2_sell = N2[-1] - n2_end
        #     N1.append(N1[-1]-n1_sell)  # ����N1(t)
        #     N2.append(N2[-1]-n2_sell)  # ����N2(t)
        #     M.append(M[-1]+n1_sell*S1[-1]+n2_sell*S2[-1]) # ����M(t)
        #     trades.append(('Sell1', n1_sell, S1[-1],'Sell2', n2_sell, S2[-1]))  # ��¼����


