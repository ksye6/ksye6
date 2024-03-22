rm(list = ls())
## remove (almost) everything in the working environment.


setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-8/R")  #set my working directory


data=read.csv("d-IBM.csv",header=T)[,7]
da=rev(data)
ibm=diff(log(da))*100
plot(ibm,xlab=" ",ylab=" ",main="Time plot of daily log returns for IBM stock",type="l",col="red",ylim=c(-28,15))
 


#Fit a TAR model

#library(TSA)



#######################################################################################
# 
#                Likelihood function
#
#######################################################################################
likfun=function(phi,data){
  n=length(data)
  y=data[3:n]
  y2=data[2:(n-1)]
  a=y-phi[1]-phi[2]*y2
  h=numeric(n-2)
  h[1]=a[1]^2
  for (i in 2:(n-2)){
    h[i]=(phi[3]+phi[4]*a[i-1]^2+phi[5]*h[i-1])*(a[i-1]<=0)+(phi[6]+phi[7]*a[i-1]^2+phi[8]*h[i-1])*(a[i-1]>0)
  }
  loglik=sum(log(h)+a^2/h)
  return(loglik)
  
}
#######################################################################################
# 
#                Estimation when r is known
#
#######################################################################################
estknow=function(data){
  initial=c(rep(0.01,8))
  fit=optim(initial,likfun,data=data,method="L-BFGS-B", 
            lower=c(rep(-0.1,2),0.001,rep(0.001,2),0.001,rep(0.001,2)),upper=c(rep(0.1,2),0.1,rep(0.99,2),0.1,rep(0.99,2)))
  estimator=fit$par
  hatpar=c(estimator)
  return(hatpar)
}

par1=estknow(ibm)
par1



#residual
data=ibm
n=length(data)
y=data[3:n]
y2=data[2:(n-1)]
a=y-par1[1]-par1[2]*y2

h=numeric(n-2)
h[1]=a[1]^2
for (i in 2:(n-2)){
  h[i]=(par1[3]+par1[4]*a[i-1]^2+par1[5]*h[i-1])*(a[i-1]<=0)+(par1[6]+par1[7]*a[i-1]^2+par1[8]*h[i-1])*(a[i-1]>0)
}

res=a/sqrt(h)

Box.test(res,10,type="Ljung") 
 
Box.test(res^2,10,type="Ljung") 
