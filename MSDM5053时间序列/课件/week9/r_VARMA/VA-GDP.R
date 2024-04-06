
###Example 8.4(page 407)
rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/Shanxi-2021/r_VARMA/ch2")  #set my working directory

#Purpose: build a simple VAR model for the monthly log returns of  IBM

 #require(MTS)  ### Load package
 da=read.table("q-gdp-ukcaus.txt",header=T)
 par(mfrow=c(2,2))
 gdp=log(da[,3:5])
 
 plot(gdp[,1],type="l",col="blue",ylab="Log rate",xlab="") 
 plot(gdp[,2],type="l",col="blue",ylab="Log rate",xlab="") 
 plot(gdp[,3],type="l",col="blue",ylab="Log rate",xlab="") 
 
 
 dim(gdp)
#[1] 126   3

 z=gdp[2:126,]-gdp[1:125,]
 z=z*100
 
 dim(z)
 
 par(mfrow=c(2,2))
 plot(z[,1],type="l",col="blue",ylab="Log rate",xlab="") 
 plot(z[,2],type="l",col="blue",ylab="Log rate",xlab="") 
 plot(z[,3],type="l",col="blue",ylab="Log rate",xlab="") 
 
 
 ccm(z, lag=10)
 
 mq(z,10)
 
 
 m1=VAR(z,2)

 resi=m1$residuals
 mq(resi,adj=18)   # plot not shown 18=2*9 (p=2 dimesion is 3)
 
 ### VAR order specification
   
  names(m2) 
  
  m2=VARorder(z)
    

    
### Recall VAR estimation
    m1=VAR(z,2)
#    names(m1)
    resi=m1$residuals
    mq(resi,adj=18)   # plot not shown 18=2*9 (p=2 dimesion is 3)
    
   
### Chi-square test for parameter constraints
   #   m3=VARchi(z,p=2)  #   Number of targeted parameters:   Chi-square test and p-value:  15.16379 0.05603778 
    
    #  m3=VARchi(z,p=2,thres=1.96)   #  Number of targeted parameters: 10  Chi-square test and p-value:  31.68739 0.000451394 
      
### Model simplification
      
     
       m1=VAR(z,2) # fit a un-constrained VAR(2) model.
       m2=refVAR(m1,thres=1.96) # Model refinement.
      
     
### Model checking

               MTSdiag(m2,adj=10)
     
## Plot not shown 
     
##### Prediction
      VARpred(m2,8)  ## Using unconstrained model     orig  125 
      
## Impulse response functions
## Using the simplified model

       Phi=m2$Phi
       Sig=m2$Sigma
       VARirf(Phi,Sig)  #   Press return to continue  ## Plots not shownSee Figures 2.8 and 2.9 of the book
       VARirf(Phi,Sig,orth=F) #      Press return to continue  ## Plots not shown
       #See Figures 2.6 and 2.7 of the book
      ### Forecast error variance decomposition
       #    original sigma  Press return to continue  ## Plots not shown
       
   