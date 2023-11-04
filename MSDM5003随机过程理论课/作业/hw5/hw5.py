import pandas as pd
import yfinance as yf
import numpy as np

#1
start_date = "2022-10-26"
end_date1 = "2023-10-26"

tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH", "^GSPC"]
tickers2 = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH"]

data1 = yf.download(tickers, start=start_date, end=end_date1)

# ��ȡÿ����Ʊ�ĵ������̼�
adj_closing_prices1 = data1["Adj Close"]

# �㵽BRK.B�����ݣ�

# Alpha Vantage: 9MV3V4OZJVR4JEN3
# Nasdaq: D1zHPDcxv3TyJRsoFVEr
# Polygon: SyBy5BXPYaNpQgvRklreByZcTvUhm0dw

import requests

# �滻Ϊ��� Polygon.io API ��Կ
api_key = 'SyBy5BXPYaNpQgvRklreByZcTvUhm0dw'

# ָ����Ʊ����
symbol = 'BRK.B'

end_date2 = "2023-10-25"

# ��������URL
url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date2}?adjusted=true&apiKey={api_key}"

# ����GET�����ȡ����
response = requests.get(url)

# ����JSON��Ӧ
data = response.json()

# ������ת��ΪDataFrame
df = pd.DataFrame(data["results"])

# ѡ��������̼���
adj_closing_prices2 = df["c"]

adj_closing_prices1["BRK.B"] = list(adj_closing_prices2)

pricedf = adj_closing_prices1.copy(deep=True)

# ��ʼ��һ��DataFrame���洢ÿ����Ʊ����������
returns = pd.DataFrame()

# ����ÿ����Ʊ����������
for ticker in tickers:
  returns[ticker] = np.log(pricedf[ticker] / pricedf[ticker].shift(1))

returns.drop(pd.to_datetime('2022-10-26'),inplace=True)

print(returns.drop(columns=['^GSPC']))

#2
pricedf0=returns.drop(columns=['^GSPC'])

# ʹ��corr������������Ծ���
correlation_matrix = pricedf0.corr()

# ������ʽ��������λС��
styled_correlation_matrix = correlation_matrix.round(3)

# ��ӡ��ʽ��������Ծ���
print(styled_correlation_matrix)


#3
# ��׼��
mean = np.mean(pricedf0, axis=0)
std = np.std(pricedf0, axis=0)
pricedf0 = (pricedf0 - mean) / std

# ��������Ծ��������ֵ����������
eigenvalues, eigenvectors = np.linalg.eig(np.cov(pricedf0.T))

#��λС�����Ȳ�������ҷֱ�������4λ��5λ��

# ��ӡ����ֵ
with np.printoptions(precision=5, suppress=True):
    print(f"Eigenvalues: {eigenvalues}")

# ��ӡ��������
with np.printoptions(precision=3, suppress=True):
    print(f"Eigenvectors: {eigenvectors}")


#4
# ʹ��argsort����������ֵ�������򲢻�ȡ����������
sorted_indices = np.argsort(eigenvalues)[::-1]

# ��������������������������ֵ����������
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
sorted_eigenvectors = sorted_eigenvectors.T

# ��ӡ����������ֵ
with np.printoptions(precision=5, suppress=True):
    print(f"Sorted Eigenvalues: {sorted_eigenvalues}")

# ��ӡ��������������
with np.printoptions(precision=3, suppress=True):
    print(f"Sorted Eigenvectors: {sorted_eigenvectors}")


#�Ա�scikit-learn��
from sklearn.decomposition import PCA

# ����PCA����
pca = PCA()

# ������ݲ��������ɷַ���
pca.fit(pricedf0)

# �鿴���ɷַ����Ľ��
explained_variance_ratio = pca.explained_variance_ratio_  # ���ɷֵķ�����ͱ���
components = pca.components_  # ���ɷֵ��غɣ�����������

# ת�����ݵ����ɷֿռ�
transformed_data = pca.transform(pricedf0)

# �Ա�R
# library(reticulate)
# setwd("C:\\Users\\�����\\Desktop")
# py <- import("hw5.py")
# # ��Python�д���DataFrame��R
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

# ѡȡN=4��ǰ�������ɷ����ں����Ľ�ģ

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

# ���������ƶ���ԭ��λ��
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# ȥ��������������
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
# �ڶ��������ɷַ���Ϊ�������и���
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





