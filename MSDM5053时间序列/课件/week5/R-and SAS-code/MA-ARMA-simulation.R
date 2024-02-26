
##MA(1) model
###################################

b=arima.sim(n = 1000, list(ma=c(0.5)))
plot(b,main=" ",ylab="",xlab="",type="l")

acf(b,20,main=expression(paste("ma(1) with ", theta[1]==-0.5)),col="red")
pacf(b,20,main=expression(paste("ma(1) models with ", theta[1]==-0.5)),col="red")

Box.test(b,lag=12,type="Ljung")


m0=arima(b,order=c(0,0,1),  include.mean = FALSE)
m0

Box.test(m0$residuals,lag=12,type="Ljung")
# Box-Ljung test


pv=1-pchisq(12.98,11)
pv



#################################################################

b=arima.sim(n = 1000, list(ma=c(0.5, -0.3)))
plot(b,main=" ",ylab="",xlab="",type="l")

acf(b,20,main=expression(paste("ma(2) with ", theta[1]==0.5,  theta[2]==-0.3)),col="red")
pacf(b,20,main=expression(paste("ma(2) models with ", theta[1]==0.5, theta[2]==-0.3)),col="red")

m1=arima(b,order=c(0,0,1),include.mean = FALSE)
m1

Box.test(m1$residuals,lag=12,type="Ljung")
# Box-Ljung test


pv=1-pchisq(107.74,11)
pv


################################################################

b=arima.sim(n = 1000, list(ar=c(0.9), ma=-0.5))
plot(b,main=" ",ylab="",xlab="",type="l")
acf(b,20,main=expression(paste("ARMA "),),col="red")
pacf(b,20,main=expression(paste("ARMA "),),col="red")

m2=arima(b,order=c(1,0,1),include.mean = FALSE)
m2

Box.test(m2$residuals,lag=12,type="Ljung")
# Box-Ljung test


pv=1-pchisq(2.0424,10)
pv

##################################ARMA condition#################

b=arima.sim(n = 1000, list(ar=c(0.5), ma=-0.5))
plot(b,main=" ",ylab="",xlab="",type="l")
acf(b,20,main=expression(paste("ARMA "),),col="red")
pacf(b,20,main=expression(paste("ARMA "),),col="red")


###########################################################



###########################################################

b=arima.sim(n = 1000, list(ar=c(0.9, -0.5), ma=-0.5))
length(b)#1000
b1=b[1:990]



plot(b,main=" ",ylab="",xlab="",type="l")
acf(b,20,main=expression(paste("ARMA "),),col="red")
pacf(b,20,main=expression(paste("ARMA "),),col="red")

m3=arima(b1,order=c(2,0,1),include.mean = FALSE)
m3

Box.test(m3$residuals,lag=12,type="Ljung")
# Box-Ljung test


pv=1-pchisq(10.834,10)
pv



fore=predict(m3,10)

fore

U=append(b[990],fore$pred+1.96*fore$se)
L=append(b[990],fore$pred-1.96*fore$se)
U
L


plot(1:25,b[976:1000],ylim=c(-10,10),type="o",ylab="",xlab="",main="Forecasting")
lines(15:25,append(b[990],fore$pred),type="o",col="red")
lines(15:25, U,type="l",col="blue")
lines(15:25, L,type="l",col="blue")
#points(temp.date[2:7],p,type="o")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))



