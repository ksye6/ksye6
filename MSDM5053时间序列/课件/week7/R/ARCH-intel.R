

rm(list = ls())
## remove (almost) everything in the working environment.


#textbook page123
setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-7/R")  #set my working directory
#Purpose: build a simple ARCH model for the monthly log returns of Intel stock
da=read.table("m-intc7308.txt",header=T)[,2]

plot(da,type="l")



acf(da,20,main="",col="red",ylim=c(-0.2,1))
pacf(da,20,main="",col="red",ylim=c(-0.2,1))

Box.test(da,lag=12,type="Ljung")

at=da-mean(da)
acf(at^2,20,main="",col="red",ylim=c(-0.2,0.4))

Box.test(at^2,lag=12,type="Ljung")


#Choosing order  
pacf(at^2,20,main="",col="red",ylim=c(-0.2,0.4))

library(fGarch) # Load the package
m0=garchFit(intc~garch(3,0),data=at,trace=F)
summary(m0) # Obtain results


##alpha2 and alpha3 appear to be statistically nonsignificant at 5% level.
m1=garchFit(intc~garch(1,0),data=at,trace=F)
summary(m1) # Obtain results



# meanForecast meanError standardDeviation

#The following command fits an ARCH(1) model with Student-t dist.
m2=garchFit(intc~garch(3,0),data=at,trace=F,cond.dist="std")
summary(m2) # Output shortened.



#The following command fits an ARCH(1) model with skew Student-t dist.
# m3=garchFit(intc~garch(1,0),data=intc,cond.dist="sstd",trace=F)
# summary(m3)

# The following command fits a GARCH(1,1) model
m4=garchFit(intc~garch(1,1),data=at,trace=F)
summary(m4) # output edited.



#Next, fit an ARMA(1,0)+GARCH(1,1) model with Gaussian noises.
m5=garchFit(intc~arma(1,0)+garch(1,1),data=da,trace=F)
summary(m5)

stresi=residuals(m5,standardize=T)

plot(stresi,xlab=" ",ylab=" ",main="Time plot of daily log returns for IBM stock",type="l",col="red",ylim=c(-15,15))


Box.test(stresi,10,type="Ljung")

predict(m5, n.ahead = 10, trace = FALSE, mse = c("cond","uncond"),
        plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)
