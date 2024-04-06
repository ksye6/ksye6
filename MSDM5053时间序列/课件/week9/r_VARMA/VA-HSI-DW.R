

rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/Shanxi-2021/r_VARMA")  #set my working directory

#Purpose: build a simple VAR model for the monthly log returns of  IBM
#da=read.table("m-ibmsp2608.txt",header=T)

y1=read.csv("hsi-dji-06-09.csv",header=T)[,2]
y2=read.csv("hsi-dji-06-09.csv",header=T)[,4]

length(y1)
length(y2)

par(mfrow=c(2,2))

plot(y1,type="l" )
plot(y2,type="l")


# compute percentage log returns.
hsi=log(y1)
dow=log(y2)

#par(mfrow=c(2,2))
#layout(matrix(c(2,1), 2, 1, byrow = TRUE))
plot(hsi,type="l" )
plot(dow,type="l")


par(mfrow=c(2,2))


dh0=diff(hsi)*100
dd0=diff(dow)*100

dh=dh0[400:804]
dd=dd0[400:804]

plot(dh,type="l" )
plot(dd,type="l")


z=cbind(dh,dd) # Create a vector series
summary(z)



ccm(z, lag=14)

mq(z,10)


m1=VAR(z,4)
resi=m1$residuals

mq(resi, adj=16)   # plot not shown


### VAR order specification

names(m2) 

m2=VARorder(z)



### Recall VAR estimation
m1=VAR(z,13)
#    names(m1)
resi=m1$residuals

mq(resi,adj=52)   # plot not shown


### Chi-square test for parameter constraints
#m3=VARchi(z,p=13)  #   Number of targeted parameters:   Chi-square test and p-value:  15.16379 0.05603778 

#m3=VARchi(z,p=13,thres=1.96)   #  Number of targeted parameters: 10  Chi-square test and p-value:  31.68739 0.000451394 

### Model simplification


m1=VAR(z,13) # fit a un-constrained VAR(2) model.
m2=refVAR(m1,thres=1.96) # Model refinement.


### Model checking

MTSdiag(m2,adj=17)

## VARMA


m2=Eccm(z,maxp=14,maxq=6)      


m2=VARMA(z,p=2,q=1) ## fit a VARMA(2,1) model


MTSdiag(m2, adj=12) # Model checking

r1=m2$residuals
mq(r1,adj=12)



m3=refVARMA(m2,thres=1.96) # Model refinement.

MTSdiag(m3, adj=5) # Model checking

r1=m3$residuals
mq(r1,adj=5)




##### Prediction
VARpred(m1,8)  ## Using unconstrained model     orig  125 

## Impulse response functions
## Using the simplified model
Phi=m2$Phi
Sig=m2$Sigma
VARirf(Phi,Sig)  #   Press return to continue  ## Plots not shownSee Figures 2.8 and 2.9 of the book
VARirf(Phi,Sig,orth=F) #      Press return to continue  ## Plots not shown
#See Figures 2.6 and 2.7 of the book
### Forecast error variance decomposition
