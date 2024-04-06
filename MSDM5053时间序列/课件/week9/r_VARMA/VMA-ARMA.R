# Simulating MA  model
  
  p1=matrix(c(.816,-1.116,-.623,1.074),2,2)
  p2=matrix(c(-.643,.615,.592,-.133),2,2)
  phi=cbind(p1,p2)
  t1=matrix(c(0,-.801,-1.248,0),2,2)
  Sig=matrix(c(4,2,2,5),2,2)

  m1=VARMAsim(400,malags=c(1),theta=t1,sigma=Sig)
  zt=m1$series
  par(mfrow=c(2,2))
  plot(zt[,1],type="l",col="blue",ylab="Log rate",xlab="")
  
  plot(zt[,2],type="l",col="red",ylab="Log rate",xlab="")
  
  ccm(zt)
  
  VMAorder(zt,lag=20)
  
  m1=VMA(zt,q=1,include.mean=F)
  
  m2=VMAe(zt,q=1)
  
  MTSdiag(m1, adj=4) # Model checking
  
  r1=m1$residuals
  mq(r1,adj=4)
  
  VMApred(m1,8)
  
#Simulating ARMA model 

########################################################################### 
  
   p1=matrix(c(.816,-1.116,-.623,1.074),2,2)
   p2=matrix(c(-.643,.615,.592,-.133),2,2)
   p3=matrix(c(.2,-.6,.3,1.1),2,2) # Input phi_1
   phi=cbind(p1,p2)
   t1=matrix(c(0,-.801,-1.248,0),2,2)
   Sig=matrix(c(4,2,2,5),2,2)

   m1=VARMAsim(1000,arlags=c(1,2),malags=c(1),phi=phi,
               theta=t1,sigma=Sig)
   
   
   #m1=VARMAsim(1000,arlags=c(1, 2),malags=c(1),phi=phi,
    #            theta=t1,sigma=Sig)

   
      zt=m1$series
   
   par(mfrow=c(2,2))
   
   plot(zt[,1],type="l",col="blue",ylab="Log rate",xlab="")
   
   plot(zt[,2],type="l",col="blue",ylab="Log rate",xlab="")
   

  m2=Eccm(zt,maxp=5,maxq=6)      
    
  m2=VARMA(zt,p=2,q=1) ## fit a VARMA(2,1) model
    
  MTSdiag(m2, adj=12) # Model checking
    
  r1=m2$residuals
  mq(r1,adj=12)
    

    
#m2a=refVARMA(m2,thres=1.64) # refine the fit
    
    
m2b=refVARMA(m2,thres=1.96) # refine further the fit.

r2b=m2b$residuals
mq(r2b,adj=9)


VARMApred(m2, h=4)  
    