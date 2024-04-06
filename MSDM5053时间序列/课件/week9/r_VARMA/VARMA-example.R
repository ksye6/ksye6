

###Example 8.4(page 407)
rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/Shanxi-2021/r_VARMA/ch3")  #set my working directory

#Purpose: build a simple VAR model for the monthly log returns of  IBM
#monthly personal consumption expenditure (PCE) and disposable
#personal income (DSPI) of the United State


da1=read.table("m-pce.txt",header=T)
da2=read.table("m-dspi.txt",header=T)

par(mfrow=c(2,2))


plot(da1[,4],type="l",col="blue",ylab="Log rate",xlab="") 
plot(da2[,4],type="l",col="blue",ylab="Log rate",xlab="") 

 z1=diff(log(da1[,4]))
 z2=diff(log(da2[,4]))
 
 

 plot(z1,type="l",col="blue",ylab="Log rate",xlab="") 
 plot(z2,type="l",col="blue",ylab="Log rate",xlab="") 
 
 

 
 zt=cbind(z1,z2)*100
 colnames(zt) <- c("pceg","dspig")

  VARorder(zt)
 
  m10=VAR(zt,3) ## fit a VAR(3) model
  
  MTSdiag(m10, adj=12) ## model checking 
  
  
  m1=VAR(zt,8) ## fit a VAR(3) model

 
 m1a=refVAR(m1,thres=1.96) ## refine the VAR(3) model

 MTSdiag(m1a, adj=17) ## model checking 

 Eccm(zt,maxp=8,maxq=7)  

 m2=VARMA(zt,p=3,q=1) ## fit a VARMA(3,1) model
   
 MTSdiag(m2,adj=16)
 
 r2=m2$residuals
 
 mq(r2,adj=16)
 
 
m2a=refVARMA(m2,thres=1.64) # refine the fit
   
# m2b=refVARMA(m2,thres=1.96) # refine further the fit.
   
 MTSdiag(m2a,adj=12)
 
 r3=m2a$residuals
 mq(r3,adj=12)
 
  m2b=refVARMA(m2a,thres=1.96) # refine further the fit.
 
 MTSdiag(m2b,adj=12)
 
 r4=m2b$residuals
 mq(r4,adj=12)
 
 
  
 VARMApred(m2b,8)
 
   
   