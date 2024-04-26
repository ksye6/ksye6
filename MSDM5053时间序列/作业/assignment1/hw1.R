
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

Box.test(k,lag=12,type="Ljung",fitdf = npara) # 处理自由度，fitdf为模型参数个数

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

# 可直接检验ARCH效应检验
sp5=ts(read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//课件//week7//R//sp500.txt"))
plot(sp5)
acf(sp5,20,main="",col="red")
pacf(sp5,20,main="",col="red")
Box.test(sp5,lag=12,type="Ljung")
at=sp5-mean(sp5)
acf(at^2,20,main="",col="red")
pacf(at^2,20,main="",col="red")
Box.test(at^2,lag=12,type="Ljung")
# 若无ARCH效应, 应>0.05, 即at^2为白噪声

# ARMA模型识别
# ARMA参数估计
# ARMA参数显著性检验

# ARCH异方差检验
library(aTSA)
# arch.test(fit, output = T) # 上半残差序列及平方序列的散点图,下半PQ检验和LM检验的P值,p小拒绝原假设,具备异方差性,考虑低阶GARCH

# GARCH拟合
fit11=garch(fit$residuals, order = c(1,1))
summary(fit11)


library(fGarch)
# Fit an MA(3)+GARCH(1,1) model.
m1=garchFit(~arma(0,3)+garch(1,1),data=sp5,trace=F,cond.dist="norm")

# 显著性检验
summary(m1)

# 模型诊断
plot(fit11)

stresi=residuals(m1,standardize=T)
plot(stresi,type="l")
Box.test(stresi,10,type="Ljung")
Box.test(stresi^2,10,type="Ljung")

# 预测
predict(m1, n.ahead = 10, trace = FALSE, mse = c("cond","uncond"), plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)

# 其他GARCH族 https://www.math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/fts-garch.html#garch-garchm
library(rugarch)
spec1=ugarchspec(variance.model=list(model="eGARCH"), mean.model=list(armaOrder=c(0,0),include.mean = TRUE),distribution.model = "std")
# GARCH-M model: sGARCH, archm=TRUE, ARCH-in-mean parameter: c(archm);
# EGARCH model: eGARCH, leverage parameter: α_1, leverage = α_1/γ_1;
# IGARCH model: iGARCH; ...
mm=ugarchfit(spec=spec1,data=data)
res=residuals(mm,standardize=T)
Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925
Box.test(res^2,10,type="Ljung")#p-value = 0.01082
# 预测
forecast = ugarchforecast(mm, n.ahead = 4, data=data)
plot(forecast, which = 1)

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


##################################################################################################################  VARMA
library(mvtnorm)
p1=matrix(c(.816,-1.116,-.623,1.074),2,2)
p2=matrix(c(-.643,.615,.592,-.133),2,2)
p3=matrix(c(.2,-.6,.3,1.1),2,2) # Input phi_1
phi=cbind(p1,p2)
t1=matrix(c(0,-.801,-1.248,0),2,2)
Sig=matrix(c(4,2,2,5),2,2)

# 平稳处理
# MTSplot(y)

# Use MTS Package
library(MTS)
### VARMA
m1=VARMAsim(1000,arlags=c(1,2),malags=c(1),phi=phi,theta=t1,sigma=Sig)
zt=m1$series
par(mfrow=c(2,2))
plot(zt[,1],type="l",col="blue",ylab="Log rate",xlab="")
plot(zt[,2],type="l",col="blue",ylab="Log rate",xlab="")
# 白噪声检验
ccm(zt, lag=10)
mq(zt,adj=12)
# 模型识别 + 参数估计
m2=Eccm(zt,maxp=5,maxq=6)
m2=VARMA(zt,p=2,q=1) ## fit a VARMA(2,1) model
# 残差检验
MTSdiag(m2, adj=12) # adj=自由度缩减个数=非0参数个数
# or mq
r1=m2$residuals
mq(r1,adj=12) # 系数矩阵的总参数个数
# 参数显著性检验 + 残差检验
m2b=refVARMA(m2,thres=1.96) # refine further the fit.
MTSdiag(m2b, adj=10)
# or mq
r2b=m2b$residuals
mq(r2b,adj=10)
# 预测
VARMApred(m2b, h=4)

### VAR
m1=VARMAsim(1000,arlags=c(1,2),phi=phi,sigma=Sig)
zt=m1$series
# 白噪声检验
ccm(zt, lag=10)
mq(zt,10)
# 模型识别 + 参数估计
VARorder(zt, maxp = 10, output = T) # VARselect(zt, lag.max=10,type="const")
m2=VAR(zt,p=2) # ,output = T, include.mean =T, fixed = NULL
# 残差检验
MTSdiag(m2,adj=8)
# or mq
resi=m2$residuals
mq(resi,adj=8) # 系数矩阵的总参数个数
# 参数显著性检验 + 残差检验
m2b=refVAR(m2,thres=1.96) # Model refinement.
MTSdiag(m2b,adj=8)
# or mq
mq(m2b$residuals,adj=8)
# 预测
VARpred(m2b,8)

### VMA
m1=VARMAsim(1000,malags=c(1),theta=t1,sigma=Sig)
zt=m1$series
# 白噪声检验
ccm(zt, lag=10)
mq(zt,10)
# 模型识别 + 参数估计
VMAorder(zt,lag=20)
m2=VMA(zt,q=1) # ,include.mean=F
# m2=VMAe(zt,q=1)
# 残差检验
MTSdiag(m2,adj=4)
# or mq
resi=m2$residuals
mq(resi,adj=4) # 系数矩阵的总参数个数
# 参数显著性检验 + 残差检验
m2b=refVMA(m2,thres=1.96) # Model refinement.  # refVMAe()
MTSdiag(m2b,adj=4)
# or mq
mq(m2b$residuals,adj=4)
# 预测
VMApred(m2b,2)
# ls("package:MTS")
# help(package = "MTS")
# https://github.com/d-/MTS/blob/master/R/MTS.R

# 或者用VARMA拟合VMA，设置参数p为0

VMApred <- function(model,h=1,orig=0){
  # Computes the i=1, 2, ..., h-step ahead predictions of a VMA(q) model.
  #
  # model is a VMA output object.
  # created on April 20, 2011
  #
  x=model$data
  resi=model$residuals
  Theta=model$Theta
  sig=model$Sigma
  mu=model$mu
  q=model$MAorder
  np=dim(Theta)[2]
  psi=-Theta
  #
  nT=dim(x)[1]
  k=dim(x)[2]
  if(orig <= 0)orig=nT
  if(orig > T)orig=nT
  if(length(mu) < 1)mu=rep(0,k)
  if(q > orig){
    cat("Too few data points to produce forecasts","\n")
  }
  pred=NULL
  se=NULL
  px=as.matrix(x[1:orig,])
  for (j in 1:h){
    fcst=mu
    t=orig+j
    for (i in 1:q){
      jdx=(i-1)*k
      t1=t-i
      if(t1 <= orig){
        theta=Theta[,(jdx+1):(jdx+k)]
        fcst=fcst-matrix(resi[t1,],1,k)%*%t(theta)
      }
    }
    px=rbind(px,fcst)
    #
    Sig=sig
    if (j > 1){
      jj=min(q,(j-1))
      for (ii in 1:jj){
        idx=(ii-1)*k
        wk=psi[,(idx+1):(idx+k)]
        Sig=Sig+wk%*%sig%*%t(wk)
      }
    }
    se=rbind(se,sqrt(diag(Sig)))
  }
  cat("Forecasts at origin: ",orig,"\n")
  print(px[(orig+1):(orig+h),],digits=4)
  cat("Standard Errors of predictions: ","\n")
  print(se[1:h,],digits=4)
  pred=px[(orig+1):(orig+h),]
  if(orig < nT){
    cat("Observations, predicted values, and errors","\n")
    tmp=NULL
    jend=min(nT,(orig+h))
    for (t in (orig+1):jend){
      case=c(t,x[t,],px[t,],x[t,]-px[t,])
      tmp=rbind(tmp,case)
    }
    colnames(tmp) <- c("time",rep("obs",k),rep("fcst",k),rep("err",k))
    idx=c(1)
    for (j in 1:k){
      idx=c(idx,c(0,1,2)*k+j+1)
    }
    tmp = tmp[,idx]
    print(tmp,digits=3)
  }
  
  VMApred <- list(pred=pred,se.err=se)
}





























