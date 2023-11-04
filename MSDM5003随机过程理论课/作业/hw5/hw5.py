import pandas as pd
import yfinance as yf
import numpy as np

#1
start_date = "2022-10-26"
end_date1 = "2023-10-26"

tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH", "^GSPC"]
tickers2 = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH"]

data1 = yf.download(tickers, start=start_date, end=end_date1)

# 获取每个股票的调整收盘价
adj_closing_prices1 = data1["Adj Close"]

# 搞到BRK.B的数据：

# Alpha Vantage: 9MV3V4OZJVR4JEN3
# Nasdaq: D1zHPDcxv3TyJRsoFVEr
# Polygon: SyBy5BXPYaNpQgvRklreByZcTvUhm0dw

import requests

# 替换为你的 Polygon.io API 密钥
api_key = 'SyBy5BXPYaNpQgvRklreByZcTvUhm0dw'

# 指定股票代码
symbol = 'BRK.B'

end_date2 = "2023-10-25"

# 构建请求URL
url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date2}?adjusted=true&apiKey={api_key}"

# 发起GET请求获取数据
response = requests.get(url)

# 解析JSON响应
data = response.json()

# 将数据转换为DataFrame
df = pd.DataFrame(data["results"])

# 选择调整收盘价列
adj_closing_prices2 = df["c"]

adj_closing_prices1["BRK.B"] = list(adj_closing_prices2)

pricedf = adj_closing_prices1.copy(deep=True)

# 初始化一个DataFrame来存储每个股票的日收益率
returns = pd.DataFrame()

# 计算每个股票的日收益率
for ticker in tickers:
  returns[ticker] = np.log(pricedf[ticker] / pricedf[ticker].shift(1))

returns.drop(pd.to_datetime('2022-10-26'),inplace=True)

print(returns.drop(columns=['^GSPC']))

#2
pricedf0=returns.drop(columns=['^GSPC'])

# 使用corr方法计算相关性矩阵
correlation_matrix = pricedf0.corr()

# 创建样式来保留三位小数
styled_correlation_matrix = correlation_matrix.round(3)

# 打印样式化的相关性矩阵
print(styled_correlation_matrix)


#3
# 标准化
mean = np.mean(pricedf0, axis=0)
std = np.std(pricedf0, axis=0)
pricedf0 = (pricedf0 - mean) / std

# 计算相关性矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(np.cov(pricedf0.T))

#三位小数精度不够因此我分别设置了4位和5位：

# 打印特征值
with np.printoptions(precision=5, suppress=True):
    print(f"Eigenvalues: {eigenvalues}")

# 打印特征向量
with np.printoptions(precision=3, suppress=True):
    print(f"Eigenvectors: {eigenvectors}")


#4
# 使用argsort函数对特征值降序排序并获取排序后的索引
sorted_indices = np.argsort(eigenvalues)[::-1]

# 根据排序后的索引重新排列特征值和特征向量
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
sorted_eigenvectors = sorted_eigenvectors.T

# 打印排序后的特征值
with np.printoptions(precision=5, suppress=True):
    print(f"Sorted Eigenvalues: {sorted_eigenvalues}")

# 打印排序后的特征向量
with np.printoptions(precision=3, suppress=True):
    print(f"Sorted Eigenvectors: {sorted_eigenvectors}")


#对比scikit-learn库
from sklearn.decomposition import PCA

# 创建PCA对象
pca = PCA()

# 拟合数据并进行主成分分析
pca.fit(pricedf0)

# 查看主成分分析的结果
explained_variance_ratio = pca.explained_variance_ratio_  # 主成分的方差解释比例
components = pca.components_  # 主成分的载荷（特征向量）

# 转换数据到主成分空间
transformed_data = pca.transform(pricedf0)

# 对比R
# library(reticulate)
# setwd("C:\\Users\\张铭韬\\Desktop")
# py <- import("hw5.py")
# # 从Python中传递DataFrame到R
# df <- py$pricedf0
# pca <- prcomp(df)
# summary(pca)


#5
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10), dpi=300)
plt.plot(range(0,10), sorted_eigenvalues, marker='o')
plt.xlabel('Eigenmode')
plt.ylabel('Eigenvalue')

plt.show()

#7
cumulative_variance=np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
cumulative_variance

# 选取N=4即前三个主成分用于后续的建模

#8

fig, ax = plt.subplots(figsize=(12, 9))

np.random.seed(1)

for i, ticker in enumerate(tickers2):
  plt.scatter(components[1][i], components[2][i], color=np.random.rand(3,), marker='o')

plt.xlabel('2nd')
plt.ylabel('3rd')
plt.title('2nd - 3rd PCA plot')

legend = plt.legend(tickers2, loc='upper left', title='names')
plt.setp(legend.get_title(), fontsize=9)

# plt.axis('off')

# 将坐标轴移动到原点位置
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# 去掉上面和右面的线
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.xticks(np.arange(-0.5, 1, 0.25))
plt.yticks(np.arange(-0.5, 1, 0.25))
plt.grid(True)

plt.show()


#A

#9
names=tickers2.copy()
weight_EWS=np.array([0.1]*10)
price=pricedf.drop(columns=['^GSPC']).iloc[0].values
volume_EWS=weight_EWS/price

Series_EWS=np.zeros(len(pricedf))
for k in range(len(pricedf)):
  Series_EWS[k]=np.dot(pricedf.drop(columns=['^GSPC']).iloc[k],volume_EWS)


#B

#10
SD=np.array(np.std(returns.drop(columns=['^GSPC']), axis=0))
inverse=1/SD
weight_RP=inverse/sum(inverse)
volume_RP=weight_RP/price

Series_RP=np.zeros(len(pricedf))
for k in range(len(pricedf)):
  Series_RP[k]=np.dot(pricedf.drop(columns=['^GSPC']).iloc[k],volume_RP)

#C

#11
equal_wt=np.array([0.1]*10)
Projection_4=np.dot(components[0:4],equal_wt)
Projection_4

#12
# 第二三四主成分符号为负，进行更改
Projection_4[1]=-Projection_4[1]
Projection_4[2]=-Projection_4[2]
Projection_4[3]=-Projection_4[3]
Projection_4
components[1]=-components[1]
components[2]=-components[2]
components[3]=-components[3]
components[0:4]

stock_wt_EWP=np.mean(components[0:4], axis=0)
stock_wt_EWP

#13
stock_wt_EWP_norm = stock_wt_EWP/np.sum(stock_wt_EWP)
volume_EWP=stock_wt_EWP_norm/price

Series_EWP=np.zeros(len(pricedf))
for k in range(len(pricedf)):
  Series_EWP[k]=np.dot(pricedf.drop(columns=['^GSPC']).iloc[k],volume_EWP)


#D

#14
Eigenvalue=sorted_eigenvalues[0:4]
Risk_wt=1/np.sqrt(Eigenvalue)

stock_wt_DRP=np.zeros(10)
for j in range(10):
  stock_wt_DRP[j]=np.dot(components[0:4][:,j], Risk_wt)/sum(Risk_wt)

stock_wt_DRP_norm=stock_wt_DRP/np.sum(stock_wt_DRP)
volume_DRP=stock_wt_DRP_norm/price

Series_DRP=np.zeros(len(pricedf))
for k in range(len(pricedf)):
  Series_DRP[k]=np.dot(pricedf.drop(columns=['^GSPC']).iloc[k],volume_DRP)


#15
Series_HIS=np.zeros(len(pricedf))
for k in range(len(pricedf)):
  Series_HIS[k]=pricedf["^GSPC"].iloc[k]/pricedf["^GSPC"].iloc[0]


fig,ax=plt.subplots(figsize=(12,9), dpi=300)

x=np.arange(251)
ax.plot(x, Series_EWS, label='EWS', color='blue')
ax.plot(x, Series_EWP, label='EWP', color='green')
ax.plot(x, Series_RP, label='RP', color='red')
ax.plot(x, Series_DRP, label='DRP', color='purple')
ax.plot(x, Series_HIS, label='HIS', color='black')


ax.set_xlabel('Day')
ax.set_ylabel('Value')
ax.legend()

ax.set_aspect(160)
plt.grid(True)

plt.show()


#16
data = {
    'EWS': Series_EWS,
    'RP': Series_RP,
    'EWP': Series_EWP,
    'DRP': Series_DRP,
    'HIS': Series_HIS
}

df = pd.DataFrame(data)

Gain=df.iloc[250]/df.iloc[0]
SD_mean=df.std()/df.mean()
mini=df.min()

df.loc["Gain"]=Gain
df.loc["SD/mean"]=SD_mean
df.loc["Minimum"]=mini

print(df.iloc[251:254])


#17
df1=pd.DataFrame(data)

beta=np.zeros(5)
for i in range(5):
  beta[i]=np.cov(df1.iloc[:,i], df1.iloc[:,4])[0,1]/np.var(df1.iloc[:,4])

df.loc["beta"]=beta
print(df.iloc[251:255])





