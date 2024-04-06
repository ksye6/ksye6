

###Example 8.4(page 407)
rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/Shanxi-2021/r_VARMA/ch3")  #set my working directory

#Purpose: build a simple VAR model for the monthly log returns of  IBM


 da=read.table("m-dec15678-6111.txt",header=T)
 head(da)
 

  x=log(da[,2:6]+1)*100
  rtn=cbind(x$dec5,x$dec8)

  
  plot(x$dec5,type="l",col="blue",ylab="Log rate",xlab="")
  
  plot(x$dec8,type="l",col="red",ylab="Log rate",xlab="")
  
   #tdx=c(1:612)/12+1961
#   par(mfcol=c(2,1))
   ccm(rtn)

    VMAorder(rtn,lag=20) # Command for identifying MA order   Q(j,m) Statistics
                         #m=20  
     m1=VMA(rtn,q=1)
     
     MTSdiag(m1)
     
      r1=m1$residuals
      mq(r1,adj=4) ## Adjust the degrees of freedom
     
      VMApred(m1,8)
      
      
    ###################################################################  
      
       m2=VMAe(rtn,q=1)  
       
       MTSdiag(m2) # Model checking
       
       r2=m2$residuals
       mq(r2,adj=4)
 
 
#############################################################################      
 
 
 da=read.table("m-ibmko-0111.txt",header=T)
 head(da)
 
 x=da[,2:3]*100
 
 
 plot(x[,1],type="l",col="blue",ylab="Log rate",xlab="")
 
 plot(x[,2],type="l",col="blue",ylab="Log rate",xlab="")


  mq(x,10)
  
  m1=VMA(x,q=1,include.mean=F)
  
  m2=VMAe(x,q=1,include.mean=F)
  
  
  
  MTSdiag(m2) # Model checking
  
  r2=m2$residuals
  mq(r2,adj=4)
  
  


