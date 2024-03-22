rm(list = ls())
## remove (almost) everything in the working environment.


setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-8/R")  #set my working directory


data=read.csv("d-IBM.csv",header=T)[,7]
da=rev(data)
ibm=diff(log(da))*100
plot(ibm,xlab=" ",ylab=" ",main="Time plot of daily log returns for IBM stock",type="l",col="red",ylim=c(-28,15))

Box.test(ibm,10,type="Ljung")

Box.test(ibm^2,10,type="Ljung")

library(fGarch)
m=garchFit(ibm~arma(0,0)+garch(1,1),data=ibm,trace=F)
summary(m)

#Obtain standardized residuals.
stresi=residuals(m,standardize=T)

plot(stresi,xlab=" ",ylab=" ",main="Time plot of daily residual for IBM stock",type="l",col="red",ylim=c(-15,15))



Box.test(stresi,10,type="Ljung")

Box.test(stresi,20,type="Ljung")

Box.test(stresi^2,10,type="Ljung")

Box.test(stresi^2,20,type="Ljung")



#The unconditional mean of the model: 0.061196
#sample mean: 0.0378514
mean(ibm)
#indicating that the model might be misspecified

predict(m, n.ahead = 100, trace = FALSE, mse = c("cond","uncond"),
        plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)


####################################################################

library(rugarch)

spec1=ugarchspec(variance.model=list(model="eGARCH"),
                 mean.model=list(armaOrder=c(0,0),include.mean = TRUE) )


mm=ugarchfit(spec=spec1,data=ibm)
mm  ### see output

#plot(mm)

res=residuals(mm,standardize=T)


Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925

Box.test(res^2,10,type="Ljung")#p-value = 0.01082

#predict(mm, n.ahead = 10, trace = FALSE, mse = c("cond","uncond"),
        plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)
#




