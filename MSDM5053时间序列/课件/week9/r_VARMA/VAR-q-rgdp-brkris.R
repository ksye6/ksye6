
rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/Shanxi-2021/r_VARMA/ch2")  #set my working directory
#Purpose: build a simple VAR model for the monthly log returns of  IBM

da=read.table("q-rgdp-brkris.txt",header=T)

head(da)

# compute percentage log returns.
x=log(da[,3:5]/100+1)*100
par(mfrow=c(2,2))

plot(x$Brazil,type="l",col="blue",ylab="Log rate",xlab="")

plot(x$Korea,type="l",col="black",ylab="Log rate",xlab="")

plot(x$Israel,type="l",col="red",ylab="Log rate",xlab="")


rtn=cbind(x$Brazil, x$Korea, x$Israel)
rtn


ccm(rtn, lag=10)

mq(rtn,10)


m1=VARorder(rtn)

m2=VAR(rtn,1)

resi=m2$residuals

mq(resi, adj=9)   # plot not shown


#m3=VARchi(rtn,p=1,thres=1.96)   #  Number of targeted parameters: 6


m1=VAR(rtn,2) # fit a un-constrained VAR(1) model.
m2=refVAR(m1,thres=1.96) # Model refinement.

resi=m2$residuals

### Model checking

mq(resi,adj=7)   #

MTSdiag(m2,adj=7)


##### Prediction
VARpred(m2,8)  ## Using unconstrained model 

## Impulse response functions Using the simplified model

Phi=m2$Phi
Sig=m2$Sigma
#VARirf(Phi,Sig)  #   Press return to continue  
VARirf(Phi,Sig,orth=F) #      Press return to continue  ## Plots not shown

