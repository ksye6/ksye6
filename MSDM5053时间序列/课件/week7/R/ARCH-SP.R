

rm(list = ls())
## remove (almost) everything in the working environment.


#textbook page123
setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-7/R")  #set my working directory


#########Example3.3 page134
#library(fGarch)
sp5=ts(read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//课件//week7//R//sp500.txt"))
plot(sp5,type="l")

acf(sp5,20,main="",col="red",ylim=c(-1,1))

pacf(sp5,20,main="",col="red",ylim=c(-1,1))

Box.test(sp5,lag=12,type="Ljung")

at=sp5-mean(sp5)
acf(at^2,20,main="",col="red",ylim=c(-1,1))


pacf(sp5^2,20,main="",col="red",ylim=c(-1,1))

Box.test(sp5^2,lag=12,type="Ljung")


# Fit an MA(3)+GARCH(1,1) model.
m1=garchFit(~arma(0,3)+garch(1,1),data=sp5,trace=F)
summary(m1)


predict(m1, n.ahead = 10, trace = FALSE, mse = c("cond","uncond"),
        plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)

# Below, fit an AR(3)+GARCH(1,1) model.
m1=garchFit(~arma(3,0)+garch(1,1),data=sp5,trace=F)
summary(m1)


m2=garchFit(~garch(1,1),data=sp5,trace=F)
summary(m2)

# Below, fit a GARCH(1,1) model with Student-t distribution.
#m2=garchFit(~garch(1,1),data=sp5,trace=F,cond.dist="std")

#Obtain standardized residuals.
stresi=residuals(m2,standardize=T)
plot(stresi,type="l")
Box.test(stresi,10,type="Ljung")
Box.test(stresi^2,10,type="Ljung")

predict(m2,5)
