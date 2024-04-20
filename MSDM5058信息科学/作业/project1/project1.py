import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

### 1 Data Preprocessing 
ticker1 = "MA"  # Mastercard
ticker2 = "MSFT"  # Microsoft

data = yf.download([ticker1, ticker2], start="2009-01-01", end="2024-04-11")

# S(t)
cp_MA = data["Close"][ticker1][241:3842]
cp_MSFT = data["Close"][ticker2][241:3842]

# X(t)
ts_MA = np.log(np.array(cp_MA[1:]) / np.array(cp_MA[:-1]))
ts_MSFT = np.log(np.array(cp_MSFT[1:]) / np.array(cp_MSFT[:-1]))

ts_MA_update = ts_MA - ts_MA.mean()
ts_MSFT_update = ts_MSFT - ts_MSFT.mean()

# Plot S(t)
plt.figure(figsize=(12, 12))
plt.plot(cp_MA, label=ticker1)
plt.plot(cp_MSFT, label=ticker2)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Time Series")
plt.legend()
plt.show()

# Plot X(t)
plt.figure(figsize=(12, 12))
plt.plot(ts_MA_update, label=ticker1)
plt.xlabel("Time")
plt.ylabel("Daily Returns")
plt.title("Daily Returns of MA")
plt.legend()
plt.subplots_adjust(left=0.18)
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(ts_MSFT_update, label=ticker2)
plt.xlabel("Time")
plt.ylabel("Daily Returns")
plt.title("Daily Returns of MSFT")
plt.legend()
plt.subplots_adjust(left=0.18)
plt.show()


### 2 Stationarity and Autocorrelation 
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from pmdarima.arima import auto_arima

# Perform the augmented Dickey-Fuller test on X to judge its stationarity
result_MA = adfuller(ts_MA_update)
print('Result of ADF test for ts_MA_update:')
print('ADF Statistic:', result_MA[0]) #Large -ve stats --> reject null, time series stationary
print('p-value:', result_MA[1]) #p-value smaller than 0.05 --> reject null, time series stationary
print('used lag:', result_MA[2]) #No. of lags used
print('critical values: ', result_MA[4]) #Critical values at 1%, 5%, 10%

result_MSFT = adfuller(ts_MSFT_update)
print('Result of ADF test for ts_MSFT_update:')
print('ADF Statistic:', result_MSFT[0]) #Large -ve stats --> reject null, time series stationary
print('p-value:', result_MSFT[1]) #p-value smaller than 0.05 --> reject null, time series stationary
print('used lag:', result_MSFT[2]) #No. of lags used
print('critical values: ', result_MSFT[4]) #Critical values at 1%, 5%, 10%

# Plot the ACF and PACF of X
plt.figure(figsize=(12, 12))
plot_acf(ts_MA_update)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.ylim(-0.5,0.5)
plt.title("ACF of Daily Returns for " + ticker1)
plt.show()

plt.figure(figsize=(12, 12))
plot_pacf(ts_MA_update)
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.ylim(-0.5,0.5)
plt.title("PACF of Daily Returns for " + ticker1)
plt.show()
# MA: ARMA(1,1)
# bestfit:
model1 = auto_arima(ts_MA_update, start_p=0, start_q=0, seasonal=False, max_p=5, max_q=5, trace=True)

plt.figure(figsize=(12, 12))
plot_acf(ts_MSFT_update)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("ACF of Daily Returns for " + ticker2)
plt.show()

plt.figure(figsize=(12, 12))
plot_pacf(ts_MSFT_update)
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.title("PACF of Daily Returns for " + ticker2)
plt.show()
# MSFT: ARMA(1,1)
# bestfit:
model2 = auto_arima(ts_MSFT_update, start_p=0, start_q=0, seasonal=False, max_p=5, max_q=5, trace=True)

# Plot the ACF and PACF of |X|
plt.figure(figsize=(12, 12))
plot_acf(abs(ts_MA_update))
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("ACF of Daily Returns for " + ticker1)
plt.subplots_adjust(left=0.18)
plt.show()

plt.figure(figsize=(12, 12))
plot_pacf(abs(ts_MA_update))
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.title("PACF of Daily Returns for " + ticker1)
plt.subplots_adjust(left=0.18)
plt.show()

plt.figure(figsize=(12, 12))
plot_acf(abs(ts_MSFT_update))
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("ACF of Daily Returns for " + ticker2)
plt.subplots_adjust(left=0.18)
plt.show()

plt.figure(figsize=(12, 12))
plot_pacf(abs(ts_MSFT_update))
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.title("PACF of Daily Returns for " + ticker2)
plt.subplots_adjust(left=0.18)
plt.show()

# Absolute value eliminates the direction of returns and focuses only on the magnitude of the change. It captures the dependence 
# between the magnitude of X and its past magnitudes.
# Also it helps identify non-linear patterns in data.


### 3 Fractal Behaviour of Time Series
## 3.1 Hurst Exponent 
import nolds
from sklearn.linear_model import LinearRegression
np.random.seed(1234)
H1 = nolds.hurst_rs(ts_MA_update)
print(H1)
# H1=0.451 < 0.5, the ts_MA has a bit of negative effect, indicating that the future trend is opposite to the past.

np.random.seed(111)
H2 = nolds.hurst_rs(ts_MSFT_update)
print(H2)
# H2=0.481 < 0.5, the ts_MA has a bit of negative effect, indicating that the future trend is opposite to the past.

# from hurst import compute_Hc
# compute_Hc(ts_MA_update,kind='change')
# compute_Hc(ts_MSFT_update,kind='change')

# N:窗口数; n:每个窗口的样本数
# def rescaled_range(X, n):
#     # 计算每个窗口内的均值
#     X_mean = np.mean(X.reshape(-1, n), axis=1)
#     # 计算相对于均值的累积偏差
#     deviation = np.cumsum(X.reshape(-1, n) - X_mean[:, np.newaxis], axis=1)
#     # 计算每个窗口内的范围
#     range_ = np.max(deviation, axis=1) - np.min(deviation, axis=1)
#     # 计算每个窗口内的标准差
#     std = np.std(X.reshape(-1, n), axis=1)
#     # 计算重标定范围
#     return np.mean(range_ /std)

def compute_RS(X, n):
    L = len(X)
    num_windows = L // n
    # 将序列分割为不重叠的窗口
    windows = np.array([X[i*n:(i+1)*n] for i in range(num_windows)])
    # 计算每个窗口的R/S值
    rs_values = np.zeros(num_windows)
    for i in range(num_windows):
        window = windows[i]
        cumsum = np.cumsum(window - np.mean(window))
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(window)
        rs_values[i] = R / S
    # 计算R/S指数
    rs_ratio = np.mean(rs_values)
    return rs_ratio

# 计算不同窗口大小的重标定范围
window_sizes = np.array([10,20,30,40,50,60,80,90,120,180,360])

# MA
rescaled_ranges_MA = [compute_RS(ts_MA_update, n) for n in window_sizes]
RS_exponent1 = np.polyfit(np.log(window_sizes), np.log(rescaled_ranges_MA), 1)[0]

plt.figure(figsize=(12, 12))
plt.plot(window_sizes, rescaled_ranges_MA, color='b',linewidth=2, label='R(n)')
plt.plot(window_sizes, np.power(window_sizes, H1), color='r', label='n^' + str(round(H1,5)), linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('window size n')
plt.ylabel('rescaled range R(n)')
plt.title('Recalibration range analysis of ts_MA')
plt.legend()
plt.grid(True)
plt.show()

# MSFT
rescaled_ranges_MSFT = [compute_RS(ts_MSFT_update, n) for n in window_sizes]
RS_exponent2 = np.polyfit(np.log(window_sizes), np.log(rescaled_ranges_MSFT), 1)[0]

plt.figure(figsize=(12, 12))
plt.plot(window_sizes, rescaled_ranges_MSFT, color='b',linewidth=2, label='R(n)')
plt.plot(window_sizes, np.power(window_sizes, H2), color='r', label='n^' + str(round(H2,5)), linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('window size n')
plt.ylabel('rescaled range R(n)')
plt.title('Recalibration range analysis of ts_MSFT')
plt.legend()
plt.grid(True)
plt.show()

## 3.2 Detrended Fluctuation Analysis (DFA)
def DFA(X,n):
    L = len(X)//n
    nf = int(L*n)
    
    y = np.cumsum(X - np.mean(X))
    y_hat = []
    for i in range(int(L)):
        x = np.arange(1,n+1,1)
        y_temp = y[int(i*n+1)-1:int((i+1)*n)]
        coef = np.polyfit(x,y_temp,1)
        y_hat.append(np.polyval(coef,x))
    fn = np.sqrt(sum((np.asarray(y)-np.asarray(y_hat).reshape(-1))**2)/nf)
    return fn

# MA
DFA_MA = [DFA(ts_MA_update, n) for n in window_sizes]
DFA_exponent1 = np.polyfit(np.log(window_sizes)[:-2],np.log(DFA_MA)[:-2],1)[0]

plt.figure(figsize=(12, 12))
plt.plot(window_sizes, DFA_MA, color='b', label='F(n) ~ n^'+str(round(DFA_exponent1,5)), linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('window size n')
plt.ylabel('detrend fluctuations F(n)')
plt.title('DFA of ts_MA')
plt.legend()
plt.grid(True)
plt.show()

# nolds.dfa(ts_MA_update)

# MSFT
DFA_MSFT = [DFA(ts_MSFT_update, n) for n in window_sizes]
DFA_exponent2 = np.polyfit(np.log(window_sizes)[:-2],np.log(DFA_MSFT)[:-2],1)[0]

plt.figure(figsize=(12, 12))
plt.plot(window_sizes, DFA_MSFT, color='b', label='F(n) ~ n^'+str(round(DFA_exponent2,5)), linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('window size n')
plt.ylabel('detrend fluctuations F(n)')
plt.title('DFA of ts_MSFT')
plt.legend()
plt.grid(True)
plt.show()

# nolds.dfa(ts_MSFT_update)

# Both are consistent with the Hurst exponent.

## 3.3 Multifractality

def calculate_M(q, tau_values, Y):
    N = len(Y)
    M = np.zeros(tau_values.shape)
    for i, t in enumerate(tau_values):
        if t >= 0:
            M[i] = np.mean(np.abs(Y[t:] - Y[:N-t]) ** q)
        else:
            M[i] = np.mean(np.abs(Y[:N+t] - Y[-t:]) ** q)
    return M

q_values = np.linspace(1, 5, 9)
tau_values = np.arange(1, 50)

# MA
Y1 = np.array(np.log(cp_MA))
M_values1 = np.zeros((len(q_values), len(tau_values)))
for i, q in enumerate(q_values):
    M_values1[i] = calculate_M(q, tau_values, Y1)

plt.figure(figsize=(12, 12))
for i, q in enumerate(q_values):
    plt.plot(tau_values, (M_values1[i] ** (1/q)), label=f'q={q:.2f}')
plt.xlabel('τ')
plt.ylabel('M(q,τ)^(1/q)')
plt.title('M(q,τ)^(1/q) of ts_MA')
plt.legend()
plt.grid(True)
plt.show()

# MSFT
Y2 = np.array(np.log(cp_MSFT))
M_values2 = np.zeros((len(q_values), len(tau_values)))
for i, q in enumerate(q_values):
    M_values2[i] = calculate_M(q, tau_values, Y2)

plt.figure(figsize=(12, 12))
for i, q in enumerate(q_values):
    plt.plot(tau_values, (M_values2[i] ** (1/q)), label=f'q={q:.2f}')
plt.xlabel('τ')
plt.ylabel('M(q,τ)^(1/q)')
plt.title('M(q,τ)^(1/q) of ts_MSFT')
plt.legend()
plt.grid(True)
plt.show()

## M(q,τ)^(1/q) ~ τ^(f(q)/q)
from scipy.optimize import curve_fit
def power_law(tau, k, alpha):
    return  k * (tau ** alpha)


# MA
fq_q_list1 = []
for i in range(len(q_values)):
    popt, pcov = curve_fit(power_law, tau_values[:15], M_values1[i][:15])
    k = popt[0]
    alpha = popt[1]
    fq_q_list1.append(alpha)

plt.figure(figsize=(12, 12))
plt.plot(q_values, fq_q_list1)
plt.xlabel('q')
plt.ylabel('f(q)/q')
plt.title('f(q)/q of ts_MA')
plt.grid(True)
plt.show()

print('f(1)='+str(fq_q_list1[0]))
# when q = 1, f(1) ≈ 0.472, which is a little bit higher than H1 = 0.451, but is still less than 0.5 so there exists multifractal.

# MSFT
fq_q_list2 = []
for i in range(len(q_values)):
    popt, pcov = curve_fit(power_law, tau_values[:15], M_values2[i][:15])
    k = popt[0]
    alpha = popt[1]
    fq_q_list2.append(alpha)

plt.figure(figsize=(12, 12))
plt.plot(q_values, fq_q_list2)
plt.xlabel('q')
plt.ylabel('f(q)/q')
plt.title('f(q)/q of ts_MSFT')
plt.grid(True)
plt.show()

print('f(1)='+str(fq_q_list2[0]))
# when q = 1, f(1) ≈ 0.479, which is a little bit smaller than H2 = 0.481, but is still less than 0.5 so there exists multifractal.

# MDFA
def MDFA(X,q,n):
    L = len(X)//n
    nf = int(L*n)
    
    y = np.cumsum(X - np.mean(X))
    y_hat = []
    for i in range(int(L)):
        x = np.arange(1,n+1,1)
        y_temp = y[int(i*n+1)-1:int((i+1)*n)]
        coef = np.polyfit(x,y_temp,1)
        y_hat.append(np.polyval(coef,x))
    fn = (sum((np.asarray(y)-np.asarray(y_hat).reshape(-1))**q)/nf)**(1/q)
    return fn

q_values = np.array([1,2,4,6,8,10])

# MA
MDFA_values1 = np.zeros((len(q_values), len(window_sizes)))
for i, q in enumerate(q_values):
    MDFA_values1[i] = [MDFA(ts_MA_update,q,n) for n in window_sizes]

alpha_q_list1 = []
for i in range(len(q_values)):
    popt, pcov = curve_fit(power_law, window_sizes, MDFA_values1[i])
    k = popt[0]
    alpha = popt[1]
    alpha_q_list1.append(alpha)

plt.figure(figsize=(12, 12))
plt.plot(q_values, alpha_q_list1)
plt.xlabel('q')
plt.ylabel('α(q) in MDFA')
plt.title('α(q) in MDFA of ts_MA')
plt.grid(True)
plt.show()

# MSFT
MDFA_values2 = np.zeros((len(q_values), len(window_sizes)))
for i, q in enumerate(q_values):
    MDFA_values2[i] = [MDFA(ts_MSFT_update,q,n) for n in window_sizes]

alpha_q_list2 = []
for i in range(len(q_values)):
    popt, pcov = curve_fit(power_law, window_sizes, MDFA_values2[i])
    k = popt[0]
    alpha = popt[1]
    alpha_q_list2.append(alpha)

plt.figure(figsize=(12, 12))
plt.plot(q_values, alpha_q_list2)
plt.xlabel('q')
plt.ylabel('α(q) in MDFA')
plt.title('α(q) in MDFA of ts_MSFT')
plt.grid(True)
plt.show()

# Still both are consistent with the ones obtained before.


### 4 Granger Causality
from statsmodels.tsa.statespace.varmax import VARMAX
merged_data = pd.concat([pd.Series(ts_MA_update,name='MA'), pd.Series(ts_MSFT_update,name='MSFT')], axis=1)
train_size = int(len(merged_data))-3
train_data, test_data = merged_data[:train_size], merged_data[train_size:]
 
best_aic = np.inf
best_order = None
best_model = None
pq_range = range(2) # 取值范围
for p in pq_range:
    for q in pq_range:
        try:
            model = VARMAX(train_data, order=(p, q))
            result = model.fit()
            aic = result.aic
            if aic < best_aic:
                best_aic = aic
                best_order = (p, q)
                best_model = result
        except:
            continue

print("Best order:", best_order)
print("Best AIC:", best_aic)
coefficients = best_model.params
print(coefficients)
best_model.summary()
# intercept.MA and intercept.MSFT are not significant. Other coefficients' p-values are < 0.05.

from statsmodels.tsa.stattools import grangercausalitytests
causality_result = grangercausalitytests(train_data, maxlag=20, verbose=True)
# when p-value < 0.05, we refuse H0, there exists Granger Causality at corresponding lags.


### 5 Fourier Transform and Power Spectrum

# MA
# 计算傅里叶变换
fourier_coeffs_MA = np.fft.fft(ts_MA_update)

# 计算频率轴
N = len(ts_MA_update)
sampling_rate = 1
freq = np.fft.fftfreq(N, d=1/sampling_rate)

# 绘制幅度随频率的变化图
plt.figure(figsize=(12, 12))
plt.plot(freq, fourier_coeffs_MA)
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('Fourier Transform of ts_MA')
plt.grid(True)
plt.show()

# MSFT
# 计算傅里叶变换
fourier_coeffs_MSFT = np.fft.fft(ts_MSFT_update)

# 绘制幅度随频率的变化图
plt.figure(figsize=(12, 12))
plt.plot(freq, fourier_coeffs_MSFT)
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('Fourier Transform of ts_MSFT')
plt.grid(True)
plt.show()

# PSD
# MA
# plt.figure(figsize=(12, 12))
# frequencies, psd = plt.psd(ts_MA_update, NFFT=3600, Fs=2)
# # Plot the PSD
# plt.xlabel('Frequency[Hz]')
# plt.ylabel('PSD [V**2/Hz] (linear)')
# plt.title('Power Spectral Density of ts_MA')
# plt.show()

from scipy import signal
plt.figure(figsize=(12, 12))
f1, Pxx_den1 = signal.periodogram(ts_MA_update, 2)
plt.plot(f1, Pxx_den1)
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title('Power Spectral Density of ts_MA')
plt.subplots_adjust(left=0.18)
plt.show()

# MSFT
# plt.figure(figsize=(12, 12))
# frequencies, psd = plt.psd(ts_MSFT_update, NFFT=3600, Fs=2)
# # Plot the PSD
# plt.xlabel('Frequency[Hz]')
# plt.ylabel('PSD [V**2/Hz] (linear)')
# plt.title('Power Spectral Density of ts_MSFT')
# plt.show()

plt.figure(figsize=(12, 12))
f2, Pxx_den2 = signal.periodogram(ts_MSFT_update, 2)
plt.plot(f2, Pxx_den2)
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title('Power Spectral Density of ts_MSFT')
plt.subplots_adjust(left=0.18)
plt.show()


### 6 Empirical Mode Decomposition
from PyEMD.EMD import EMD
from PyEMD.visualisation import Visualisation
xx = np.linspace(1., 3600, 3600, endpoint=False)

#MA
emd1 = EMD()
IMFs_MA = emd1(ts_MA_update)
# 绘制分解后的IMFs
k = len(IMFs_MA)  # IMF的数量
selected_IMFs_MA = [1, int(k/4), int(k/2), int(3*k/4), k]  # 选择要绘制的IMFs
# 绘制原始数据
plt.figure(figsize=(12, 12))
plt.subplot(len(selected_IMFs_MA) + 1, 1, 1)
plt.plot(ts_MA_update, label='Original Data')
plt.legend(fontsize='small',loc='upper right')
plt.title('IMFs of ts_MA from EMD')
# 绘制IMFs
for i, imf in enumerate(IMFs_MA):
    if (i+1) in selected_IMFs_MA:
        plt.subplot(len(selected_IMFs_MA) + 1, 1, selected_IMFs_MA.index(i+1) + 2)
        plt.plot(imf, label='IMF {}'.format(i+1))
        plt.legend(fontsize='small',loc='upper right')
plt.subplots_adjust(hspace=0.5)
plt.show()

#MSFT
emd2 = EMD()
IMFs_MSFT = emd2(ts_MSFT_update)
# 绘制分解后的IMFs
k2 = len(IMFs_MSFT)  # IMF的数量
selected_IMFs_MSFT = [1, int(k2/4), int(k2/2), int(3*k2/4), k2]  # 选择要绘制的IMFs
# 绘制原始数据
plt.figure(figsize=(12, 12))
plt.subplot(len(selected_IMFs_MSFT) + 1, 1, 1)
plt.plot(ts_MSFT_update, label='Original Data')
plt.legend(fontsize='small',loc='upper right')
plt.title('IMFs of ts_MSFT from EMD')
# 绘制IMFs
for i, imf in enumerate(IMFs_MSFT):
    if (i+1) in selected_IMFs_MSFT:
        plt.subplot(len(selected_IMFs_MSFT) + 1, 1, selected_IMFs_MSFT.index(i+1) + 2)
        plt.plot(imf, label='IMF {}'.format(i+1))
        plt.legend(fontsize='small',loc='upper right')
plt.subplots_adjust(hspace=0.5)
plt.show()


from hurst import compute_Hc
# MA
he1 = []
for i, imf in enumerate(IMFs_MA):
    H, c, val = compute_Hc(imf,kind='change')
    he1.append(H)

orders1 = np.arange(1, len(IMFs_MA) + 1)
plt.figure(figsize=(12, 12))
plt.plot(orders1, he1, marker='o')
plt.xlabel('Order')
plt.ylabel('Hurst Exponent')
plt.title('Hurst Exponent of IMFs_MA')
plt.show()

# MSFT
he2 = []
for i, imf in enumerate(IMFs_MSFT):
    H, c, val = compute_Hc(imf,kind='change')
    he2.append(H)

orders2 = np.arange(1, len(IMFs_MSFT) + 1)
plt.figure(figsize=(12, 12))
plt.plot(orders2, he2, marker='o')
plt.xlabel('Order')
plt.ylabel('Hurst Exponent')
plt.title('Hurst Exponent of IMFs_MSFT')
plt.show()

# A linear gradual downward trend


# MA first2 IMFs' PSD
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
f_MA_1, Pxx_den_MA_1 = signal.periodogram(IMFs_MA[0], 2)
plt.plot(f_MA_1, Pxx_den_MA_1)
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MA's 1st IMF")

plt.subplot(2, 1, 2)
f_MA_2, Pxx_den_MA_2 = signal.periodogram(IMFs_MA[1], 2)
plt.plot(f_MA_2, Pxx_den_MA_2)
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MA's 2nd IMF")

plt.subplots_adjust(hspace=0.5,left=0.18)
plt.show()

# MSFT first2 IMFs' PSD
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
f_MSFT_1, Pxx_den_MSFT_1 = signal.periodogram(IMFs_MSFT[0], 2)
plt.plot(f_MSFT_1, Pxx_den_MSFT_1)
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MSFT's 1st IMF")

plt.subplot(2, 1, 2)
f_MSFT_2, Pxx_den_MSFT_2 = signal.periodogram(IMFs_MSFT[1], 2)
plt.plot(f_MSFT_2, Pxx_den_MSFT_2)
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MSFT's 2nd IMF")

plt.subplots_adjust(hspace=0.5,left=0.18)
plt.show()

# 1st IMF's PSD concentrates on low frequency range; 2nd IMF's PSD concentrates on high frequency range.


# MA
MA_res1 = ts_MA_update - IMFs_MA[0]
MA_res2 = ts_MA_update - IMFs_MA[0] - IMFs_MA[1]

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(signal.periodogram(ts_MA_update, 2)[0], signal.periodogram(ts_MA_update, 2)[1])
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title('Power Spectral Density of ts_MA')

plt.subplot(3, 1, 2)
plt.plot(signal.periodogram(MA_res1, 2)[0], signal.periodogram(MA_res1, 2)[1])
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MA-IMF1")

plt.subplot(3, 1, 3)
plt.plot(signal.periodogram(MA_res2, 2)[0], signal.periodogram(MA_res2, 2)[1])
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MA-IMF1-IMF2")

plt.subplots_adjust(hspace=0.7,left=0.18)
plt.show()

# MSFT
MSFT_res1 = ts_MSFT_update - IMFs_MSFT[0]
MSFT_res2 = ts_MSFT_update - IMFs_MSFT[0] - IMFs_MSFT[1]

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(signal.periodogram(ts_MSFT_update, 2)[0], signal.periodogram(ts_MSFT_update, 2)[1])
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title('Power Spectral Density of ts_MSFT')

plt.subplot(3, 1, 2)
plt.plot(signal.periodogram(MSFT_res1, 2)[0], signal.periodogram(MSFT_res1, 2)[1])
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MSFT-IMF1")

plt.subplot(3, 1, 3)
plt.plot(signal.periodogram(MSFT_res2, 2)[0], signal.periodogram(MSFT_res2, 2)[1])
plt.xlabel('Frequency[Hz]')
plt.ylabel('PSD [V**2/Hz] (linear)')
plt.title("Power Spectral Density of ts_MSFT-IMF1-IMF2")

plt.subplots_adjust(hspace=0.7,left=0.18)
plt.show()

# The higher order IMFs minused by original signal, the lower frequency range remaining signal concentrates at.















