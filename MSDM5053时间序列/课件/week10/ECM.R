phi=matrix(c(0.5,-0.25,-1.0,0.5),2,2); theta=matrix(c(0.2,-0.1,-0.4,0.2),2,2)
Sig=diag(2)
mm=VARMAsim(300,arlags=c(1),malags=c(1),phi=phi,theta=theta,sigma=Sig)

zt=mm$series[,c(2,1)]


beta=matrix(c(1,0.5),2,1)
m1=ECMvar(zt,3,ibeta=beta)
names(m1)



plot(zt[,2], type="l", col="red", ylim=c(-50,50))
lines(zt[,1],type="l")

u=zt[,1]+0.506*zt[,2]
u

plot(u, type="l", col="red", ylim=c(-5,5))


MTSdiag(m1, gof = 10, adj = 0, level = T)




###Example 8.4(page 407)
rm(list = ls())

#library(vars)

###############################################################################################
#Example 8.6(page425):demonstrate VARMA model
y1=read.csv("GS1.csv",header=T)[,2]
y3=read.csv("GS3.csv",header=T)[,2]

length(y3)

y=cbind(y1,y3)

MTSplot(y)

#To ensure the positiveness of U.S. interest rates, we analyze the log series.
lgy=cbind(log(y1),log(y3))

plot(lgy[,1],type="l",col="blue",ylab="Log rate",xlab="",ylim=c(-0.3,3))
lines(lgy[,2],type="s",col="red")

VARorder(lgy, maxp = 10, output = T)


#choose p=4

#Performs conditional maximum likelihood estimation of a VARMA model. Multivariate Gaussian likelihood function is used
fit5=VARMA(lgy, p=2, q=1, include.mean = T)


fit6=refVARMA(fit5)


MTSdiag(fit6, gof = 10, adj = 0, level = T)
#level:	Logical switch for printing residual cross-correlation matrices

  
#Computes the forecasts of a VAR model, the associated standard errors of forecasts and the mean squared errors of forecasts
VARMApred(fit6, h=4)



beta=matrix(c(1,-1.0),2,1)

m2=ECMvar(lgy,4,ibeta=beta)

MTSdiag(m2, gof = 10, adj = 0, level = T)

cy=lgy[,1]-0.957*lgy[,2]

plot(cy, type="l", col="red", ylim=c(-1,1))

