# File Format: gbk
# Please click File -> Reopen with Encoding... -> CP936

library(quantmod)

############## 研究香港各大巴/地铁公司的股价情况。选取的长度为满足白噪声检验的长度

# 冠忠
setSymbolLookup(KC=list(name="0306.HK",src="yahoo"))
getSymbols("KC", from = "2021-01-01", to = "2024-04-20")
KC=na.omit(KC)
KC
KC=KC$"0306.HK.Adjusted"
plot(KC)

# 港铁
setSymbolLookup(MTR=list(name="0066.HK",src="yahoo"))
getSymbols("MTR", from = "2015-01-01", to = "2024-04-20")
MTR=na.omit(MTR)
MTR
MTR=MTR$"0066.HK.Adjusted"
plot(MTR)


# 载通
setSymbolLookup(TI=list(name="0062.HK",src="yahoo"))
getSymbols("TI", from = "2015-01-01", to = "2024-04-20")
TI=na.omit(TI)
TI
TI=TI$"0062.HK.Adjusted"
plot(TI)


############################################# KC
library(forecast)
# 差分+平稳性检验
KC_dr = diff(KC) / lag(KC)
KC_dr = na.omit(KC_dr)
plot(KC_dr)

ndiffs(KC_dr) # 差分0次
length(KC_dr)

library(tseries)
pp.test(KC_dr) # < 0.05 则平稳

# 白噪声检验
for( i in c(2,5,9,11) ){
  print(Box.test(KC_dr,lag=i,type="Ljung-Box"))
} # < 0.05 则非白噪声, 有意义

# ARCH效应检验
KC_dr_at=KC_dr-mean(KC_dr)
acf(KC_dr_at^2,20,main="",col="red")
pacf(KC_dr_at^2,20,main="",col="red")
for( i in c(2,5,9,11) ){
  print(Box.test(KC_dr_at^2,lag=i,type="Ljung-Box"))
} # 若无ARCH效应, 应>0.05
# 存在ARCH效应

# ARMA模型识别
acf(KC_dr)
pacf(KC_dr)
library(TSA)
eacf(KC_dr)
auto.arima(KC_dr,trace = T)

Arima(KC_dr,order = c(0,0,1), include.drift = T)$aic
Arima(KC_dr,order = c(0,0,2), include.drift = T)$aic
Arima(KC_dr,order = c(1,0,1), include.drift = T)$aic
Arima(KC_dr,order = c(1,0,2), include.drift = T)$aic
Arima(KC_dr,order = c(2,0,1), include.drift = T)$aic
Arima(KC_dr,order = c(2,0,2), include.drift = T)$aic

# ARMA参数估计
# 使用forecast包里的Arima(), 包含漂移项的影响
KC_md = Arima(KC_dr,order = c(0,0,1), include.drift = T)

# ARMA参数显著性检验
# t统计量
t = abs(KC_md$coef)/sqrt(diag(KC_md$var.coef))
# 自由度
df_t = length(KC_dr)-length(KC_md$coef)
# pt()
pt(t,df_t,lower.tail = F)
# p<0.05, 则显著
# intercept,ma2,drift不显著

# # 残差检验
# library(stats)
# tsdiag(KC_md)

# # ARCH效应验证
# library(aTSA)
# tent = arima(KC_dr,order = c(0,0,1), method = 'ML')
# arch.test(tent, output = T) # 上半残差序列及平方序列的散点图,下半PQ检验和LM检验的P值,p小拒绝原假设,具备异方差性,考虑低阶GARCH

# ARMA-EGARCH 拟合
library(fGarch)
library(rugarch)
KC_spec=ugarchspec(variance.model=list(model="eGARCH",garchOrder = c(1, 1)),
                   mean.model=list(armaOrder=c(0,1),include.mean = TRUE),
                   distribution.model = "sstd")

KC_md_2=ugarchfit(spec=KC_spec,data=KC_dr)
KC_md_2  ### 去除不显著部分alpha1

KC_spec=ugarchspec(variance.model=list(model="eGARCH",garchOrder = c(1, 1)),
                   mean.model=list(armaOrder=c(0,1),include.mean = TRUE),
                   distribution.model = "sstd", fixed.pars = c(alpha1 = 0))

KC_md_2=ugarchfit(spec=KC_spec,data=KC_dr)
KC_md_2

# model: ARMA(0,1)-EGARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0 - θ_1 * a_t-1
# a_t = σ_t * ε_t
# ln[(σ_t)^2] = α_0 + [α_1*(ε_t-1) + γ_1(|ε_t-1| - E|ε_t-1|)] + β_1*ln[(σ_t-1)^2]
# where μ_0 = -0.001123, θ_1 = -0.080011, α_0 = -1.220427, α_1 = 0, β_1 = 0.808370, γ_1 = 1.00000
# skew > 0, 分布右偏; shape < 3, 扁平
# α_1 = 0, 不存在杠杆效应

plot(KC_md_2, which = 10)
plot(KC_md_2, which = 11)

KC_stresi=residuals(KC_md_2,standardize=T)
plot(KC_stresi,type="l")
Box.test(KC_stresi,808,type="Ljung-Box",fitdf = 4) # p-value > 0.05, white noise
Box.test(KC_stresi^2,808,type="Ljung-Box",fitdf = 4) # p-value > 0.05, remains no ARCH effect

# 模型预测
forecast = ugarchforecast(KC_md_2, n.ahead = 3, data=KC_dr)
plot(forecast, which = 1)
plot(forecast, which = 3)


############################################# MTR
# 差分+平稳性检验
MTR_dr = diff(MTR) / lag(MTR)
MTR_dr = na.omit(MTR_dr)
plot(MTR_dr)

ndiffs(MTR_dr) # 差分一次
MTR_dr_1 = diff(MTR_dr)
MTR_dr_1 = na.omit(MTR_dr_1)
length(MTR_dr_1)

pp.test(MTR_dr_1) # < 0.05 则平稳

# 白噪声检验
for( i in c(2,5,9,11) ){
  print(Box.test(MTR_dr_1,lag=i,type="Ljung-Box"))
} # < 0.05 则非白噪声, 有意义

# ARCH效应检验
MTR_dr_1_at=MTR_dr_1-mean(MTR_dr_1)
acf(MTR_dr_1_at^2,20,main="",col="red")
pacf(MTR_dr_1_at^2,20,main="",col="red")
for( i in c(2,5,9,11) ){
  print(Box.test(MTR_dr_1_at^2,lag=i,type="Ljung-Box"))
} # 若无ARCH效应, 应>0.05
# 存在ARCH效应

# ARMA模型识别
acf(MTR_dr_1)
pacf(MTR_dr_1)
eacf(MTR_dr_1)

Arima(MTR_dr_1,order = c(0,0,1), include.drift = T)$aic
Arima(MTR_dr_1,order = c(0,0,2), include.drift = T)$aic
Arima(MTR_dr_1,order = c(1,0,2), include.drift = T)$aic

# ARMA参数估计
# 使用forecast包里的Arima(), 不包含漂移项的影响
MTR_md = Arima(MTR_dr_1,order = c(0,0,1), include.drift = F)

# ARMA参数显著性检验
# t统计量
t = abs(MTR_md$coef)/sqrt(diag(MTR_md$var.coef))
# 自由度
df_t = length(MTR_dr_1)-length(MTR_md$coef)
# pt()
pt(t,df_t,lower.tail = F)
# p<0.05, 则显著
# intercept不显著

# # 残差检验
# library(stats)
# tsdiag(MTR_md)

# # ARCH效应验证
# library(aTSA)
# tent = arima(MTR_dr_1,order = c(0,0,1), method = 'ML')
# arch.test(tent, output = T) # 上半残差序列及平方序列的散点图,下半PQ检验和LM检验的P值,p小拒绝原假设,具备异方差性,考虑低阶GARCH

# ARMA-EGARCH 拟合
MTR_spec=ugarchspec(variance.model=list(model="eGARCH",garchOrder = c(1, 1)),
                   mean.model=list(armaOrder=c(0,1),include.mean = TRUE),
                   distribution.model = "sstd")

MTR_md_2=ugarchfit(spec=MTR_spec,data=MTR_dr_1)
MTR_md_2  ### 去除不显著部分mu和alpha1

MTR_spec=ugarchspec(variance.model=list(model="eGARCH",garchOrder = c(1, 1)),
                    mean.model=list(armaOrder=c(0,1),include.mean = FALSE),
                    distribution.model = "sstd", fixed.pars = c(mu = 0, alpha1 = 0))

MTR_md_2=ugarchfit(spec=MTR_spec,data=MTR_dr_1)
MTR_md_2

# model: ARMA(0,1)-EGARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0 - θ_1 * a_t-1
# a_t = σ_t * ε_t
# ln[(σ_t)^2] = α_0 + [α_1*(ε_t-1) + γ_1(|ε_t-1| - E|ε_t-1|)] + β_1*ln[(σ_t-1)^2]
# where μ_0 = 0, θ_1 = -0.99060, α_0 = -0.21897, α_1 = 0, β_1 = 0.97578, γ_1 = 0.18286
# skew > 0, 分布右偏; shape > 3, 尖峰
# α_1 = 0, 不存在杠杆效应

plot(MTR_md_2, which = 10)
plot(MTR_md_2, which = 11)

MTR_stresi=residuals(MTR_md_2,standardize=T)
plot(MTR_stresi,type="l")
Box.test(MTR_stresi,2288,type="Ljung-Box",fitdf = 4) # p-value > 0.05, white noise
Box.test(MTR_stresi^2,2288,type="Ljung-Box",fitdf = 4) # p-value > 0.05, remains no ARCH effect

# 模型预测
forecast = ugarchforecast(MTR_md_2, n.ahead = 3, data=MTR_dr_1)
plot(forecast, which = 1)
plot(forecast, which = 3)

############################################# TI
# 差分+平稳性检验
TI_dr = diff(TI) / lag(TI)
TI_dr = na.omit(TI_dr)
plot(TI_dr)

ndiffs(TI_dr) # 差分一次
TI_dr_1 = diff(TI_dr)
TI_dr_1 = na.omit(TI_dr_1)
length(TI_dr_1)

pp.test(TI_dr_1) # < 0.05 则平稳

# 白噪声检验
for( i in c(2,5,9,11) ){
  print(Box.test(TI_dr_1,lag=i,type="Ljung-Box"))
} # < 0.05 则非白噪声, 有意义

# ARCH效应检验
TI_dr_1_at=TI_dr_1-mean(TI_dr_1)
acf(TI_dr_1_at^2,20,main="",col="red")
pacf(TI_dr_1_at^2,20,main="",col="red")
for( i in c(2,5,9,11) ){
  print(Box.test(TI_dr_1_at^2,lag=i,type="Ljung-Box"))
} # 若无ARCH效应, 应>0.05
# 存在ARCH效应

# ARMA模型识别
acf(TI_dr_1)
pacf(TI_dr_1)
eacf(TI_dr_1)

Arima(TI_dr_1,order = c(0,0,1), include.drift = T)$aic
Arima(TI_dr_1,order = c(0,0,3), include.drift = T)$aic
Arima(TI_dr_1,order = c(1,0,3), include.drift = T)$aic
Arima(TI_dr_1,order = c(2,0,3), include.drift = T)$aic


# ARMA参数估计
# 使用forecast包里的Arima(), 不包含漂移项的影响
TI_md = Arima(TI_dr_1,order = c(0,0,1), include.drift = F)

# ARMA参数显著性检验
# t统计量
t = abs(TI_md$coef)/sqrt(diag(TI_md$var.coef))
# 自由度
df_t = length(TI_dr_1)-length(TI_md$coef)
# pt()
pt(t,df_t,lower.tail = F)
# p<0.05, 则显著
# intercept不显著

# # 残差检验
# library(stats)
# tsdiag(TI_md)

# # ARCH效应验证
# library(aTSA)
# tent = arima(TI_dr_1,order = c(0,0,1), method = 'ML')
# arch.test(tent, output = T) # 上半残差序列及平方序列的散点图,下半PQ检验和LM检验的P值,p小拒绝原假设,具备异方差性,考虑低阶GARCH

# ARMA-EGARCH 拟合
TI_spec=ugarchspec(variance.model=list(model="eGARCH",garchOrder = c(1, 1)),
                    mean.model=list(armaOrder=c(0,1),include.mean = TRUE),
                    distribution.model = "sstd")

TI_md_2=ugarchfit(spec=TI_spec,data=TI_dr_1)
TI_md_2  ### 去除不显著部分mu和alpha1

TI_spec=ugarchspec(variance.model=list(model="eGARCH",garchOrder = c(1, 1)),
                    mean.model=list(armaOrder=c(0,1),include.mean = FALSE),
                    distribution.model = "sstd", fixed.pars = c(mu = 0, alpha1 = 0))

TI_md_2=ugarchfit(spec=TI_spec,data=TI_dr_1)
TI_md_2

# model: ARMA(0,1)-EGARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0 - θ_1 * a_t-1
# a_t = σ_t * ε_t
# ln[(σ_t)^2] = α_0 + [α_1*(ε_t-1) + γ_1(|ε_t-1| - E|ε_t-1|)] + β_1*ln[(σ_t-1)^2]
# where μ_0 = 0, θ_1 = -0.96347, α_0 = -0.98647, α_1 = 0, β_1 = 0.89404, γ_1 = 0.44817
# skew > 0, 分布右偏; shape > 3, 尖峰
# α_1 = 0, 不存在杠杆效应

plot(TI_md_2, which = 10)
plot(TI_md_2, which = 11)

TI_stresi=residuals(TI_md_2,standardize=T)
plot(TI_stresi,type="l")
Box.test(TI_stresi,2288,type="Ljung-Box",fitdf = 4) # p-value > 0.05, white noise
Box.test(TI_stresi^2,2288,type="Ljung-Box",fitdf = 4) # p-value > 0.05, remains no ARCH effect

# 模型预测
forecast = ugarchforecast(TI_md_2, n.ahead = 3, data=TI_dr_1)
plot(forecast, which = 1)
plot(forecast, which = 3)

############################################# 观察到 MTR 和 TI 日增长率较为接近，考察此二者的VARMA模型
library(mvtnorm)
library(MTS)

zt = as.matrix(cbind(MTR,TI))
colnames(zt) = c( "MTR", "TI")
zt = diff(log(zt))*100

plot(as.xts(zt), type="l", 
     multi.panel=TRUE, theme="white",
     main="日增长率")

# 白噪声检验
ccm(zt)
mq(zt)

# 模型识别 + 参数估计
VARorder(zt, maxp = 10, output = T)
# VMAorder(zt,lag=20)
# m2=Eccm(zt,maxp=8,maxq=6)
m2=VARMA(zt,p=7,q=0)

# 参数显著性检验 + 残差检验
m2b=refVARMA(m2,thres=1.96) # refine further the fit.
MTSdiag(m2b, adj=5)
# or mq
r2b=m2b$residuals
mq(r2b,adj=5)
# 预测
VARMApred(m2b, h=4)

















