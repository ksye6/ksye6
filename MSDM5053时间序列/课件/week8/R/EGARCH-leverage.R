rm(list = ls())
## remove (almost) everything in the working environment.


setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-7/data/")  #set my working directory


data=read.csv("HSI-06-09.csv",header=T)[,7]
#da=rev(data)
length(data)#772

plot(data,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(0,35000))


ddata=diff(log(data))*100
plot(ddata,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(-20,15))


#employ the gjrGARCH/iGARCH/eGARCH model


library(rugarch)

spec1=ugarchspec(variance.model=list(model="eGARCH"),
                 mean.model=list(armaOrder=c(0,0),include.mean = FALSE) )


mm=ugarchfit(spec=spec1,data=ddata)
mm  ### see output

#in SAS

leverage=-0.09377/0.209714
leverage

#plot(mm)

res=residuals(mm,standardize=T)


Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925

Box.test(res^2,10,type="Ljung")#p-value = 0.01082


####################################################################################


data=read.csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//课件//week7//data//Dow.csv",header=T)[,7]
#da=rev(data)
length(data)#20172
length1=20172-772
length1
data1=data[19400:20172]

plot(data1,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(0,18000))

ddata=diff(log(data1))*100
plot(ddata,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(-20,15))


#employ the gjrGARCH/iGARCH/eGARCH model


library(rugarch)

spec1=ugarchspec(variance.model=list(model="eGARCH"),
                 mean.model=list(armaOrder=c(0,0),include.mean = FALSE) )


mm=ugarchfit(spec=spec1,data=ddata)
mm  ### see output

#in SAS

leverage=-0.166607/0.127943
leverage

#plot(mm)

res=residuals(mm,standardize=T)


Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925

Box.test(res^2,10,type="Ljung")#p-value = 0.01082





####################################################################################


data=read.csv("N225-84-09.csv",header=T)[,7]
#da=rev(data)
plot(data,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(0,40000))
length(data)#6174
length1=6174-772
length1
data1=data[5402:6174]

ddata=diff(log(data1))*100
plot(ddata,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(-20,15))


#employ the gjrGARCH/iGARCH/eGARCH model


library(rugarch)

spec1=ugarchspec(variance.model=list(model="eGARCH"),
                 mean.model=list(armaOrder=c(0,0),include.mean = FALSE) )


mm=ugarchfit(spec=spec1,data=ddata)
mm  ### see output

#in SAS

leverage=-0.159876/0.078616
leverage

#plot(mm)

res=residuals(mm,standardize=T)


Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925

Box.test(res^2,10,type="Ljung")#p-value = 0.01082


#############################################################


data=read.csv("HSBC.csv",header=T)[,7]
#da=rev(data)
plot(data,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(0,160))
length(data)1573
length1=1573-772
length1
data1=data[801:1573]


ddata=diff(log(data1))*100
plot(ddata,xlab=" ",ylab=" ",main="Time plot of daily log returns",type="l",col="red",ylim=c(-20,15))


#employ the gjrGARCH/iGARCH/eGARCH model


library(rugarch)

spec1=ugarchspec(variance.model=list(model="eGARCH"),
                 mean.model=list(armaOrder=c(0,0),include.mean = FALSE) )


mm=ugarchfit(spec=spec1,data=ddata)
mm  ### see output

#in SAS

leverage=-0.128551/0.258192
leverage

#plot(mm)

res=residuals(mm,standardize=T)


Box.test(res,10,type="Ljung")#p-value = 0.9611
Box.test(res,20,type="Ljung")#p-value = 0.3925

Box.test(res^2,10,type="Ljung")#p-value = 0.01082







