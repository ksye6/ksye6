
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


################################################################################################################## ARMA
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
fore=predict(md, 10)

# 若对数化，则需要还原
exp(fore$pred+fore$se*fore$se/2)

################################################################################################################## ARIMA
# 差分+平稳性检验
set.seed(123)
k = arima.sim(500, model=list(ar=0.8,ma=0.5,order=c(1,2,1)))
ndiffs(k) # forecast
kk = diff(k)
kkk = diff(kk)
library(aTSA)
adf.test(kk)
adf.test(kkk) # < 0.05 则平稳

# 白噪声检验
for( i in c(5,9,11) ){
    print(Box.test(kkk,lag=i,type="Ljung-Box"))
} # < 0.05 则非白噪声, 有意义

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


################################################################################################################## SARIMA
# 在上述基础上先进行season阶滞后差分
# 例如diff(data,lag=n_season)
# decompose观察 plot(decompose(ts(x,frequency=nseason)))

# 差分+平稳性检验
# 白噪声检验
# 模型识别 ts函数需要设置frequency
# 参数估计
# 参数显著性检验
# 残差检验
# 模型预测

################################################################################################################## ARMA-GARCH
library(tseries)

# 平稳性检验 pp.test(difx)
# 白噪声检验
# 模型识别
# 参数估计
# 参数显著性检验

# 异方差检验
library(aTSA)
# arch.test(fit, output = T) # 上半残差序列及平方序列的散点图,下半PQ检验和LM检验的P值,p小拒绝原假设,具备异方差性,考虑低阶GARCH

# GARCH拟合
fit11=garch(fit$residuals, order = c(1,1))
summary(fit11)
# 模型诊断
plot(fit11) 

# 其他GARCH族
library(rugarch)
spec1=ugarchspec(variance.model=list(model="eGARCH"), mean.model=list(armaOrder=c(0,0),include.mean = TRUE) )
mm=ugarchfit(spec=spec1,data=ibm)
res=residuals(mm,standardize=T)
Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925
Box.test(res^2,10,type="Ljung")#p-value = 0.01082
predict(mm, n.ahead = 10, trace = FALSE, mse = c("cond","uncond"), plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)

################################################################################################################## SARIMA-EGARCH a method

# https://lbelzile.github.io/timeseRies/simulation-based-prediction-intervals-for-arima-garch-models.html

library(rugarch)
# Mean monthly temperature in UK, from Hadley center
data(CETmonthly, package = "multitaper")
# Keep period 1900-2000
mtempfull <- ts(CETmonthly[, 3], start = c(CETmonthly[1, 1], CETmonthly[1, 2]), 
                frequency = 12)
mtemp <- window(mtempfull, start = c(1900, 1), end = c(1999, 12))
months <- rep(1:12, 100)
# Fit our favorite model - Fourier basis + SARMA(3,0,1)x(1,0,1)
meanreg <- cbind(cos(2 * pi * months/12), sin(2 * pi * months/12), cos(4 * pi * months/12), sin(4 * pi * months/12))
sarima <- forecast::Arima(mtemp, order = c(3, 0, 1), seasonal = c(1, 0, 1), xreg = meanreg)
# or sarima <- forecast::Arima(mtemp, order = c(3, 0, 1), seasonal = c(1, 0, 1), xreg = fourier(mtemp, K = 2)) 遍历K取最小值
# Not great, but will serve for the illustration

# This SARIMA is basically an ARMA model of high order with constraints
# These constraints can be roughly replicated by fixing most components to
# zero For example, if we have a SARMA (0,1)x(0,1)[12], coefficients 2 to 11
# are zero We will however estimate an extra MA parameter at lag 13 which
# would normally be prod(sarima$model$theta[c(1,12)])

# Extract AR coefficients via $model$phi Extract MA coefficients via
# $model$theta Find which parameters are zero and constrain them
lnames <- c(paste0("ar", which(sapply(sarima$model$phi, function(th) {
  isTRUE(all.equal(th, 0))
}))), paste0("ma", which(sapply(sarima$model$theta, function(th) {
  isTRUE(all.equal(th, 0))
}))))
constraints <- rep(list(0), length(lnames))
names(constraints) <- lnames
order <- c(length(sarima$model$phi), length(sarima$model$theta))

# To have a non-constant variance, we can let it vary per month Could try
# with dummies, but will stick with Fourier (issues with convergence of
# optimization routine otherwise) Creating a matrix of dummies varreg <-
# model.matrix(lm(months~as.factor(months)))[,-1] Model specification
model <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(0,0), external.regressors = meanreg), 
                    mean.model = list(armaOrder = order, include.mean = TRUE, external.regressors = meanreg), distribution.model = "std", 
                    fixed.pars = constraints)
# Fit model
fitmodel <- ugarchfit(spec = model, data = mtemp)
# Coefficients of the model
fitmodel@fit$coef[fitmodel@fit$coef != 0]

# Not parsimonious, but all coefficients are significant

# Forecast K observations ahead
plotforecasted <- function(m = 60) {
  # Here, the variance depends only on the monthly index - deterministic
  # variation
  sig <- sqrt(exp(fitmodel@fit$coef["omega"] + meanreg[1:m, ] %*% coef(fitmodel)[startsWith(prefix = "vxreg", names(coef(fitmodel)))]))
  arma_par <- coef(fitmodel)[as.logical(startsWith("ar", x = names(fitmodel@fit$coef)) + startsWith("ma", x = names(fitmodel@fit$coef)))]
  # Generate replications from the ARIMA model and add the deterministic part
  simu_paths <- replicate(c(arima.sim(model = as.list(arma_par), n = m, innov = sig * rt(m, df = fitmodel@fit$coef["shape"])
                                      , start.innov = as.vector(residuals(fitmodel)), n.start = length(residuals(fitmodel)) - 1)
                            + meanreg[1:m, ] %*% coef(fitmodel)[startsWith("mxreg", x = names(fitmodel@fit$coef))] + coef(fitmodel)[1]), n = 1000)
  # Extract the sample quantiles from the Monte-Carlo replications of the paths
  confinter <- apply(simu_paths, 1, function(x) {
    quantile(x, c(0.05, 0.5, 0.95))
  })
  # Plot the series with previous years
  plot(window(mtempfull, start = c(1996, 1)), xlim = c(1996, m/12 + 2000 - 0.5), ylim = c(0, 20), 
       main = "Central England monthly temperatures", ylab = "Degrees Celsius", bty = "l")
  # Add forecasts with 90% confidence intervals
  matplot(x = seq(2000 + 1/12, by = 1/12, length.out = m), y = t(confinter), 
          lty = c(2, 1, 2), lwd = c(1, 2, 1), type = rep("l", 3), col = c(2, 1,2), add = TRUE)
}

# Plot the forecasts
plotforecasted(120)


##################################################################################################################  ????????


















