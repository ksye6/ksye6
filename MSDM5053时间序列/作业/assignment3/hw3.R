library(aTSA)
library(tseries)
library(TSA)
library(forecast)
library(fGarch)
library(rugarch)
######################### 1
df = read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//作业//assignment3//d-sbuxsp0106.txt",header=F)
#  Convert the simple returns into percentage log returns
log_returns_SBUX = log(1 + ts(df$V2))

# Stationarity test
ndiffs(log_returns_SBUX) # d=0
pp.test(log_returns_SBUX) # p-value < 0.05, reject H0, stationary

# a
# white noise test
acf(log_returns_SBUX, lag.max = 10, col="red")
Box.test(log_returns_SBUX,lag=10,type="Ljung-Box")
# p-value < 0.05, there exists serial correlation in the log returns of Starbucks stock

# b
# ARCH test
at_SBUX=log_returns_SBUX-mean(log_returns_SBUX)
acf(at_SBUX^2, lag.max = 10, col="red")
pacf(at_SBUX^2, lag.max = 10, col="red")
Box.test(at_SBUX^2,lag=10,type="Ljung-Box")
# p-value < 0.05, there exists ARCH effect in the log returns of Starbucks stock

# c
# First fit an ARMA model
auto.arima(log_returns_SBUX)
est=Arima(log_returns_SBUX, order = c(3, 0, 0))
t = abs(est$coef)/sqrt(diag(est$var.coef))
df_t = length(log_returns_SBUX)-length(est$coef)
pt(t,df_t,lower.tail = F)
# fix ar2 and ar3 and intercept to 0, which are not significant
est=Arima(log_returns_SBUX, order = c(1, 0, 0))
t = abs(est$coef)/sqrt(diag(est$var.coef))
df_t = length(log_returns_SBUX)-length(est$coef)
pt(t,df_t,lower.tail = F)
# ARMA(1,0)

m1=garchFit(log_returns_SBUX~arma(1,0)+garch(1,1),data=log_returns_SBUX,trace=F,cond.dist="norm")
summary(m1)
# All coefficients are significant.
# model: ARMA(1,0)-GARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0 + φ_1 * r_t-1
# a_t = σ_t * ε_t
# (σ_t)^2 = α_0 + α_1*(a_t-1)^2 + β_1*(σ_t-1)^2
# where μ_0 = 1.326e-03, φ_1 = -7.475e-02, α_0 = 1.472e-06, α_1 = 1.851e-02, β_1 = 9.776e-01

plot(m1, which = c(10,11,13))

stresi=residuals(m1,standardize=T)
plot(stresi,type="l")
Box.test(stresi,10,type="Ljung-Box",fitdf = 3) # p-value > 0.05, white noise
Box.test(stresi^2,10,type="Ljung-Box",fitdf = 3) # p-value > 0.05, remains no ARCH effect


######################### 2
log_returns_SP = log(1 + ts(df$V3))

# Stationarity test
ndiffs(log_returns_SP) # d=0
pp.test(log_returns_SBUX) # p-value < 0.05, reject H0, stationary

# a
# white noise test
acf(log_returns_SP, lag.max = 10, col="red")
Box.test(log_returns_SP,lag=10,type="Ljung-Box")
# p-value > 0.05, there doesn't exist any serial correlation in the log returns of S&P index

# b
# ARCH test
at_SP=log_returns_SP-mean(log_returns_SP)
acf(at_SP^2, lag.max = 10, col="red")
pacf(at_SP^2, lag.max = 10, col="red")
Box.test(at_SP^2,lag=10,type="Ljung-Box")
# p-value < 0.05, there exists ARCH effect in the log returns of S&P index

# c
spec2=ugarchspec(variance.model=list(model="iGARCH",garchOrder = c(1, 1)),
                 mean.model=list(armaOrder=c(0,0),include.mean = TRUE),
                 distribution.model = "std")

m2=ugarchfit(spec=spec2,data=log_returns_SP)
m2  ### see output
# Coefficients of mu and omega are not significant.
# model: iGARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0
# a_t = σ_t * ε_t
# (σ_t)^2 = α_0(=0) + α_1*(a_t-1)^2 + β_1*(σ_t-1)^2, where α_1 + β_1 = 1
# where μ_0 = 0.000392(not significant, can be seen as 0), α_0 = 0, α_1 = 0.062736, β_1 = 0.937264

plot(m2, which = 10)
plot(m2, which = 11)

stresi2=residuals(m2,standardize=T)
plot(stresi2,type="l")
Box.test(stresi2,10,type="Ljung-Box",fitdf = 1) # p-value > 0.05, white noise
Box.test(stresi2^2,10,type="Ljung-Box",fitdf = 1) # p-value > 0.05, remains no ARCH effect

# d
forecast = ugarchforecast(m2, n.ahead = 4, data=log_returns_SP)
plot(forecast, which = 1)
U=forecast@forecast$seriesFor+1.96*forecast@forecast$sigmaFor
L=forecast@forecast$seriesFor-1.96*forecast@forecast$sigmaFor

forecast
c(L[1],U[1])

#!! the result looks not right, I think we should take the arma model into account:
# Let's check more lags:
acf(log_returns_SP, col="red")
for( i in c(6,12,18,24,30) ){
  print(Box.test(log_returns_SP,lag=i,type="Ljung-Box"))
}
# we can see there may exist some serial correlation.
# do the ARMA model:
auto.arima(log_returns_SP)

spec2=ugarchspec(variance.model=list(model="iGARCH",garchOrder = c(1, 1)),
                 mean.model=list(armaOrder=c(3,3),include.mean = TRUE),
                 distribution.model = "std")

m2=ugarchfit(spec=spec2,data=log_returns_SP)
m2  ### see output
# Coefficients of mu and omega are not significant.
# model: ARMA(3,3)-GARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0 + φ_1 * r_t-1 + φ_2 * r_t-2 + φ_3 * r_t-3 - θ_1 * a_t-1 - θ_2 * a_t-2 - θ_3 * a_t-3
# a_t = σ_t * ε_t
# (σ_t)^2 = α_0(=0) + α_1*(a_t-1)^2 + β_1*(σ_t-1)^2, where α_1 + β_1 = 1
# where μ_0 = 0.000501, φ_1 = 0.522999, φ_2 = -1.028071, φ_3 = 0.418220, θ_1 = -0.601726, θ_2 = 1.063333, θ_3 = -0.492953
#       α_0 = 0, α_1 = 0.061252, β_1 = 0.938748

plot(m2, which = 10)
plot(m2, which = 11)

stresi2=residuals(m2,standardize=T)
plot(stresi2,type="l")
Box.test(stresi2,20,type="Ljung-Box",fitdf = 7) # p-value > 0.05, white noise
Box.test(stresi2^2,20,type="Ljung-Box",fitdf = 7) # p-value > 0.05, remains no ARCH effect

# d
forecast = ugarchforecast(m2, n.ahead = 4, data=log_returns_SP)
plot(forecast, which = 1)
# this result looks more correct

U=forecast@forecast$seriesFor+1.96*forecast@forecast$sigmaFor
L=forecast@forecast$seriesFor-1.96*forecast@forecast$sigmaFor

forecast
c(L[1],U[1])


######################### 3
# a
# fit an ARMA(1,0)-GARCH(1,1)-M model:
spec3=ugarchspec(variance.model=list(model="sGARCH",garchOrder = c(1, 1)),
                 mean.model=list(armaOrder=c(1,0),include.mean = TRUE,archm=TRUE),
                 distribution.model = "norm")

m3=ugarchfit(spec=spec3,data=log_returns_SBUX)
m3  ### see output
# model: ARMA(1,0)-GARCH(1,1)-M
# r_t = μ_t + c * (σ_t)^2 + a_t
# μ_t = μ_0 + φ_1 * r_t-1
# a_t = σ_t * ε_t
# (σ_t)^2 = α_0 + α_1*(a_t-1)^2 + β_1*(σ_t-1)^2
# where μ_0 = 0.003756, φ_1 = -0.075778, c = -0.138842, α_0 = 0, α_1 = 0.018621, β_1 = 0.977339

plot(m3, which = 10)
plot(m3, which = 11)

stresi3=residuals(m3,standardize=T)
plot(stresi3,type="l")
Box.test(stresi3,10,type="Ljung-Box",fitdf = 4) # p-value > 0.05, white noise
Box.test(stresi3^2,10,type="Ljung-Box",fitdf = 4) # p-value > 0.05, remains no ARCH effect

# b
# c = -0.138842, p-value of t-test is 0.012770 < 0.05, so the ARCH-in-mean parameter is significant.

# c
# fit an ARMA(1,0)-EGARCH(1,1) model:
spec3_2=ugarchspec(variance.model=list(model="eGARCH",garchOrder = c(1, 1)),
                   mean.model=list(armaOrder=c(1,0),include.mean = TRUE),
                   distribution.model = "norm")

m3_2=ugarchfit(spec=spec3_2,data=log_returns_SBUX)
m3_2  ### see output
# model: ARMA(1,0)-EGARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0 + φ_1 * r_t-1
# a_t = σ_t * ε_t
# ln[(σ_t)^2] = α_0 + [α_1*(ε_t-1) + γ_1(|ε_t-1| - E|ε_t-1|)] + β_1*ln[(σ_t-1)^2]
# where μ_0 = 0.000936, φ_1 = -0.076938, α_0 = -0.048337, α_1 = -0.036930, β_1 = 0.993617, γ_1 = 0.045562

plot(m3_2, which = 10)
plot(m3_2, which = 11)

stresi3_2=residuals(m3_2,standardize=T)
plot(stresi3_2,type="l")
Box.test(stresi3_2,10,type="Ljung-Box",fitdf = 4) # p-value > 0.05, white noise
Box.test(stresi3_2^2,10,type="Ljung-Box",fitdf = 4) # p-value > 0.05, remains no ARCH effect

# d
# α_1 = -0.036930, p-value of t-test is 0.000000 < 0.05, so the leverage parameter is significant.
# leverage = α_1/γ_1 = -0.036930/0.045562


######################### 4

df2 = read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//作业//assignment3//m-pg5606.txt",header=F)
#  Convert the simple returns into percentage log returns
log_returns_PG = log(1 + ts(df2$V2))

# a
# Stationarity test
ndiffs(log_returns_PG) # d=0
pp.test(log_returns_PG) # p-value < 0.05, reject H0, stationary

# white noise test
acf(log_returns_PG, lag.max = 10, col="red")
Box.test(log_returns_PG,lag=10,type="Ljung-Box")
# p-value > 0.05, there doesn't exist any serial correlation in the log returns of PG data

# ARCH test
at_PG=log_returns_PG-mean(log_returns_PG)
acf(at_PG^2, lag.max = 10, col="red")
pacf(at_PG^2, lag.max = 10, col="red")
Box.test(at_PG^2,lag=10,type="Ljung-Box")
# p-value < 0.05, there exists ARCH effect in the log returns of S&P index

# b
m4=garchFit(log_returns_PG~garch(1,1),data=log_returns_PG,trace=F,cond.dist="norm")
summary(m4)
# All coefficients are significant.
# model: GARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0
# a_t = σ_t * ε_t
# (σ_t)^2 = α_0 + α_1*(a_t-1)^2 + β_1*(σ_t-1)^2
# where μ_0 = 8.562e-03, α_0 = 8.537e-05, α_1 = 9.631e-02, β_1 = 8.624e-01

plot(m4, which = c(10,11,13))

stresi4=residuals(m4,standardize=T)
plot(stresi4,type="l")
Box.test(stresi4,10,type="Ljung-Box",fitdf = 2) # p-value > 0.05, white noise
Box.test(stresi4^2,10,type="Ljung-Box",fitdf = 2) # p-value > 0.05, remains no ARCH effect

# c
predict(m4, n.ahead = 5, trace = FALSE, mse = c("cond","uncond"), plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)
# 1 step interval:
c(-0.04930263,0.06642749)


######################### 4
df3 = read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//作业//assignment3//d-exuseu.txt",header=F)
#  Convert the simple returns into percentage log returns
log_returns_df3 = log(1 + ts(df3$V4))

# a
# Stationarity test
ndiffs(log_returns_df3) # d=1
log_returns_df3=diff(log_returns_df3)
pp.test(log_returns_PG) # p-value < 0.05, reject H0, stationary

# white noise test
acf(log_returns_df3, lag.max = 10, col="red")
Box.test(log_returns_df3,lag=10,type="Ljung-Box")
# p-value > 0.05, there doesn't exist any serial correlation in the log returns of df3 data


# b
# ARCH test
at_df3=log_returns_df3-mean(log_returns_df3)
acf(at_df3^2, lag.max = 10, col="red")
pacf(at_df3^2, lag.max = 10, col="red")
Box.test(at_df3^2,lag=10,type="Ljung-Box")
# p-value < 0.05, there exists ARCH effect in the log returns of df3 data

# c
spec5=ugarchspec(variance.model=list(model="iGARCH",garchOrder = c(1, 1)),
                 mean.model=list(armaOrder=c(0,0),include.mean = TRUE),
                 distribution.model = "norm")

m5=ugarchfit(spec=spec5,data=log_returns_df3)
m5  ### see output
# Coefficients of mu and omega are not significant.
# model: iGARCH(1,1)
# r_t = μ_t + a_t
# μ_t = μ_0
# a_t = σ_t * ε_t
# (σ_t)^2 = α_0(=0) + α_1*(a_t-1)^2 + β_1*(σ_t-1)^2, where α_1 + β_1 = 1
# where μ_0 = 0.000056(not significant, can be seen as 0), α_0 = 0, α_1 = 0.017439, β_1 = 0.982561

plot(m5, which = 10)
plot(m5, which = 11)

stresi5=residuals(m5,standardize=T)
plot(stresi5,type="l")
Box.test(stresi5,10,type="Ljung-Box",fitdf = 1) # p-value > 0.05, white noise
Box.test(stresi5^2,10,type="Ljung-Box",fitdf = 1) # p-value > 0.05, remains no ARCH effect

# d
forecast = ugarchforecast(m5, n.ahead = 4, data=log_returns_df3)
plot(forecast, which = 1)
U=forecast@forecast$seriesFor+1.96*forecast@forecast$sigmaFor
L=forecast@forecast$seriesFor-1.96*forecast@forecast$sigmaFor

forecast
c(L[1],U[1])





























