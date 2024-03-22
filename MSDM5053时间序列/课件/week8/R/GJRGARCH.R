rm(list = ls())
## remove (almost) everything in the working environment.


setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-8/R")  #set my working directory


data=read.csv("d-IBM.csv",header=T)[,7]
da=rev(data)
ibm=diff(log(da))*100
plot(ibm,xlab=" ",ylab=" ",main="Time plot of daily log returns for IBM stock",type="l",col="red",ylim=c(-28,15))
 

#employ the gjrGARCH/iGARCH/eGARCH model



library(rugarch)

spec1=ugarchspec(variance.model=list(model="gjrGARCH"),
                 mean.model=list(armaOrder=c(0,0),include.mean = FALSE) )


mm=ugarchfit(spec=spec1,data=ibm)
mm  ### see output
#plot(mm)

res=residuals(mm,standardize=T)


Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925

Box.test(res^2,10,type="Ljung")#p-value = 0.01082
 
#predict(mm, n.ahead = 10, trace = FALSE, mse = c("cond","uncond"),
#        plot=TRUE, nx=NULL, crit_val=NULL, conf=NULL)






