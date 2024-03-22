rm(list = ls())
## remove (almost) everything in the working environment.


setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-8/SAS")  #set my working directory


data=read.csv("zhu_data.csv",header=T)[,2]
length(data) #709

plot(data,xlab=" ",ylab=" ",main="Time plot of daily log returns for Oil Price",type="l",col="red",ylim=c(0,150))

oil0=diff(log(data))*100
length(oil0) #708

oil=oil0[1:704]

length(oil) #708


plot(oil,xlab=" ",ylab=" ",main="Time plot of daily log returns for Oil Price",type="l",col="red",ylim=c(-28,25))

Box.test(oil,10,type="Ljung")


acf(oil,20,main="",col="red",ylim=c(-0.2,1))
pacf(oil,20,main="",col="red",ylim=c(-0.2,1))


m3=arima(oil,order=c(4,0,0))
m3


Box.test(m3$residuals,lag=12,type="Ljung")

pv=1-pchisq(25.605,8)
pv


m4=arima(oil,order=c(0,0,8))
m4

Box.test(m4$residuals,lag=12,type="Ljung")

pv=1-pchisq(5.2298,4)
pv

m5=arima(oil,order=c(0,0,8), ,fixed=c(NA,NA,NA,0,0,0,0,NA,0))
m5

Box.test(m5$residuals,lag=12,type="Ljung")

pv=1-pchisq(9.6047,8)
pv



fore=predict(m5,4)

fore

U=append(oil[704],fore$pred+1.96*fore$se)
L=append(oil[704],fore$pred-1.96*fore$se)
U
L

plot(1:13,oil0[696:708],ylim=c(-28,25),type="o",ylab="",xlab="",main="Forecasting")
lines(9:13,append(oil[704],fore$pred),type="o",col="red")
lines(9:13, U,type="l",col="blue")
lines(9:13, L,type="l",col="blue")
#points(temp.date[2:7],p,type="o")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))


#################################################################################


Box.test(m5$residuals^2,10,type="Ljung")

Box.test(m5$residuals^2,12,type="Ljung")

Box.test(m5$residuals^2,18,type="Ljung")


library(fGarch)
m6=garchFit(oil~arma(0,8)+garch(1,1),data=oil,trace=F)
summary(m6)

#Obtain standardized residuals.
stresi=residuals(m6,standardize=T)

plot(stresi,xlab=" ",ylab=" ",main="Time plot of daily residual for oil",type="l",col="red",ylim=c(-15,15))


Box.test(stresi,10,type="Ljung")

Box.test(stresi,20,type="Ljung")

Box.test(stresi^2,10,type="Ljung")

Box.test(stresi^2,20,type="Ljung")

fore=predict(m6, n.ahead = 4, trace = FALSE, mse = c("cond","uncond"),
                      plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)

fore

L1=append(oil[704],fore$lowerInterval)
L1
U1=append(oil[704],fore$upperInterval)
U1


plot(1:13,oil0[696:708],ylim=c(-28,25),type="o",ylab="",xlab="",main="Forecasting")
lines(9:13,append(oil[704],fore$meanForecast),type="o",col="red")
lines(9:13, U1,type="l",col="blue")
lines(9:13, L1,type="l",col="blue")
#points(temp.date[2:7],p,type="o")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))



####################################################################

library(rugarch)

spec1=ugarchspec(variance.model=list(model="iGARCH"),
                 mean.model=list(armaOrder=c(0,8),include.mean = TRUE) )


mm=ugarchfit(spec=spec1,data=oil)
mm  ### see output

#plot(mm)

res=residuals(mm,standardize=T)


Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925

Box.test(res^2,10,type="Ljung")#p-value = 0.01082

#predict(mm, n.ahead = 10, trace = FALSE, mse = c("cond","uncond"),
#        plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)





