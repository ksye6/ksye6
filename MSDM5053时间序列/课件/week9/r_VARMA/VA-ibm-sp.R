
###Example 8.4(page 407)
rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/Shanxi-2021/r_VARMA")  #set my working directory

#Purpose: build a simple VAR model for the monthly log returns of  IBM
da=read.table("m-ibmsp2608.txt",header=T)


# compute percentage log returns.
ibm=(da[,2]+1)
sp5=(da[,3]+1)

par(mfrow=c(2,1))
layout(matrix(c(2,1), 2, 1, byrow = TRUE))
plot(ibm,type="l" )
plot(sp5,type="l")

z=cbind(ibm,sp5) # Create a vector series
summary(z)


ccm(z, lag=10)

mq(z,10)


m1=VAR(z,4)
 

### VAR order specification

names(m2) 

m2=VARorder(z)



### Recall VAR estimation
m1=VAR(z,5)
#    names(m1)

resi=m1$residuals
mq(resi,adj=20)   # plot not shown


### Chi-square test for parameter constraints
m3=VARchi(z,p=5)  #   Number of targeted parameters:   Chi-square test and p-value:  15.16379 0.05603778 
                  #alpha=0.1
m3=VARchi(z,p=5,thres=1.96)   #  Number of targeted parameters: 10  Chi-square test and p-value:  31.68739 0.000451394 
                              ##alpha=0.05 (number of zero parameters)
### Model simplification


m1=VAR(z,5) # fit a un-constrained VAR(2) model.

m2=refVAR(m1,thres=1.96) # Model refinement.

# thres =1.96 corrseponding to alpha=0.05

resi=m2$residuals
mq(resi,adj=8)   # plot not shown



### Model checking

MTSdiag(m2,adj=8)

## Plot not shown 

##### Prediction
VARpred(m2,8)  ## Using unconstrained model 

## Impulse response functions
## Using the simplified model
Phi=m2$Phi
Sig=m2$Sigma
VARirf(Phi,Sig)  #   Press return to continue  ## Plots not shownSee Figures 2.8 and 2.9 of the book
VARirf(Phi,Sig,orth=F) #      Press return to continue  ## Plots not shown
#See Figures 2.6 and 2.7 of the book
### Forecast error variance decomposition

#m2=refVAR(m1)  ## using default threshold

#names(m2)
#[1] "data"      "order"     "cnst"      "coef"      "aic"       "bic"      
#[7] "hq"        "residuals" "secoef"    "Sigma"     "Phi"       "Ph0"      
#Phi=m2$Phi
#Sig=m2$Sigma
#Theta=NULL
#FEVdec(Phi,Theta,Sig,lag=5)

