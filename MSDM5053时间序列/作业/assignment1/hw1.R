
# 1
# (a)

df = read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//作业//assignment1//m-dec19.txt",header=T)
Decile_1 = df[2]
acf_values = acf(Decile_1, lag.max = 24)
pacf_values = pacf(Decile_1, lag.max = 24)

# (b)
Box.test(Decile_1,lag=12,type="Ljung")
# p value is small enough to make sure the first 12 lags of ACF are not all zero.


# 2
# (a)
Decile_9 = df[3]
acf_values_2 = acf(Decile_9, lag.max = 12)

# (b)
Box.test(Decile_9,lag=12,type="Ljung")
# p value is larger than 0.05 so we accept H0: the first 12 lags of ACF are all zero.


# 3
df3 = read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//作业//assignment1//m-cpileng.txt",header=F)
Xt = df3$V4
ct = 100 * (log(Xt[2:length(Xt)]) - log(Xt[1:length(Xt)-1]))

# (a)
acf_values = acf(ct, lag.max = 12)
pacf_values = pacf(ct, lag.max = 12)
Box.test(ct,lag=12,type="Ljung")
# p value is small enough to make sure the first 12 lags of ACF are not all zero.

# (b)
zt = ct[2:length(ct)] - ct[1:length(ct)-1]
acf_values2 = acf(zt, lag.max = 12)

# (c)
model_ct = arima(ct,order=c(1,0,5))
model_ct


# 4
df4 = read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//作业//assignment1//q-gnprate.txt",header=F)

# (a)
model_q = arima(df4, order = c(3, 0, 0))
model_q

# (b)
model_q$coef
p1=c(1,-model_q$coef[1:3])
# 多项式求根
s1=polyroot(p1)
s1
# which implies the existence of stochastic cycles
Mod(s1)
k=2*pi/acos(1.542932/1.800683)
k
# the average period of business cycle is about 11.6 quarters.

# (c)
fore=predict(model_q, 4)
fore
fore$pred
fore$se


# 5

# (a)
model_D9 = arima(Decile_9, order = c(0, 0, 1))
model_D9

# (b)
# 参数显著性检验
# t统计量
t = abs(model_D9$coef)/sqrt(diag(model_D9$var.coef))
# 自由度
df_t = dim(Decile_9)[1]-length(model_D9$coef)
# pt()
pt(t,df_t,lower.tail = F)
# p<0.05, 显著

# 零均值、等方差、正态性 检验
summary(model_D9)
Box.test(model_D9$residuals,type="Ljung")
tsdiag(model_D9)
# The standardized residuals are basically distributed near the zero horizontal line and in the range of -3~3; 
# the autocorrelation function quickly drops to within the two dotted lines; 
# the P values of the Ljung-Box statistics are all greater than 0.05
# therefore, the model passes the test.

# (c)
fore=predict(model_D9, 4)
fore


######################################## ARMA
# 模型识别
# 1.eacf
library(TSA)
set.seed(123)
k = arima.sim(500, model=list(ar=0.8,ma=0.5))
eacf(k)
eacf(ts(Decile_9))
# 2.auto.arima
library(forecast)
auto.arima(ts(Decile_9))
auto.arima(k)

# 参数估计
md = arima(k,order = c(1,0,1), method = 'ML')
md

# 参数显著性检验
# t统计量
t = abs(md$coef)/sqrt(diag(md$var.coef))
# 自由度
df_t = length(k)-length(md$coef)
# pt()
pt(t,df_t,lower.tail = F)
# p<0.05, 则显著

# 残差检验
library(stats)
tsdiag(md)

# 模型预测
predict(md, 10)

######################################## ARIMA
# 差分+平稳性检验
set.seed(123)
k = arima.sim(500, model=list(ar=0.8,ma=0.5,order=c(1,2,1)))
ndiffs(k)
kk = diff(k)
kkk = diff(kk)
library(aTSA)
adf.test(kk)
adf.test(kkk) # < 0.05 则平稳

# 白噪声检验
for( i in c(5,9,11) ){
    print(Box.test(kkk,lag=i,type="Ljung-Box"))
} # < 0.05 则非白噪声

# 模型识别
# auto.arima
library(forecast)
auto.arima(k)

# 参数估计
# 使用forecast包里的Arima(), 包含漂移项的影响
md2 = Arima(k,order = c(1,2,1), include.drift = T) # Warning message: 二阶差分后不含漂移项

# 参数显著性检验
# t统计量
t = abs(md2$coef)/sqrt(diag(md2$var.coef))
# 自由度
df_t = length(k)-length(md2$coef)
# pt()
pt(t,df_t,lower.tail = F)
# p<0.05, 则显著

# 残差检验
library(stats)
tsdiag(md2)

# 模型预测
fore.gnp = forecast::forecast(md2,10) # 后四列为置信区间
plot(fore.gnp, lty=2, pch=1, type='b',xlim=c(480,512),ylim=c(20000,23500))
lines(fore.gnp$fitted, col=2, pch=2, type='b')





