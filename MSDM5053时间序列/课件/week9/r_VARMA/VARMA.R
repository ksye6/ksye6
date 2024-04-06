
###Example 8.4(page 407)
rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-9/r_VARMA/r_VARMA")  #set my working directory

#Purpose: build a simple VAR model for the monthly log returns of  IBM
da=read.table("m-ibmsp2608.txt",header=T)

# compute percentage log returns.
ibm=log(da[,2]+1)*100
sp5=log(da[,3]+1)*100

par(mfrow=c(2,1))
layout(matrix(c(2,1), 2, 1, byrow = TRUE))
plot(ibm,type="l" )
plot(sp5,type="l")

y=cbind(ibm,sp5) # Create a vector series
summary(y)


library(vars)
fit=VAR(y, lag.max=10,type="const",ic="AIC")
summary(fit)

serial.test(fit, lags.pt = 20, type = "PT.adjusted")

pre=predict(fit,n.ahead=6,ci=0.95)
pre


fit=VARselect(y, lag.max=10,type="const")
fit


#page414 For simplicity, we shall use VAR(1) specification in the demonstration.
fit1=VAR(y, lag=1,type="const")
summary(fit1)


pre=predict(fit1,n.ahead=6,ci=0.95)
#fanchart(pre)
pre

##############Use MTS Package#########################################
library(MTS)

MTSplot(y)

ph0=ccm(y, lags = 5,level=F)#page 394


#Computes information criteria and the sequential Chi-square statistics for a vector autoregressive process
VARorder(y, maxp = 13, output = T)

fit2=VAR(y, p=5, output = T, include.mean =T, fixed = NULL)
summary(fit2)



#VARMA(da, p = 0, q = 0, include.mean = T,fixed = NULL, beta=NULL, sebeta=NULL,prelim = F, details = F, thres = 2)

#Multivariate Time Series Diagnostic Checking
MTSdiag(fit2, gof = 24, adj = 0, level = F)

#Computes the forecasts of a VAR model, the associated standard errors of forecasts and the mean squared errors of forecasts
VARpred(fit2, h = 1, orig = 0, Out.level = F)


#Refine a fitted VAR model by removing simultaneously insignificant parameters
refVAR(fit2, fixed = NULL, thres = 1)

###########################################################################################
#Example 8.5(page421)

fit3=VMA(y, q=5, include.mean = T)
#Performs VMA estimation using the conditional multivariate Gaussian likelihood function
fit33=VMACpp(y, q=5, include.mean = T)
#This isthe same function as VMA, with the likelihood function implemented in C++ for efficiency.

# Output
# Coefficient(s):
#      Estimate  Std. Error  t value Pr(>|t|)    
# ibm   1.08948     0.22842    4.770 1.85e-06 ***
# sp5   0.42995     0.18625    2.308 0.020972 *  
#       0.02633     0.04129    0.638 0.523750    
#      -0.15098     0.05234   -2.885 0.003918 ** 
#      -0.08649     0.04130   -2.094 0.036269 *  
#       0.14998     0.05243    2.860 0.004230 ** 
#      -0.05280     0.04303   -1.227 0.219876    
#       0.11898     0.05277    2.254 0.024171 *  
#       0.01776     0.04010    0.443 0.657949    
#       0.04265     0.05383    0.792 0.428154    
#       0.04027     0.04093    0.984 0.325184    
#      -0.13713     0.05336   -2.570 0.010170 *  
#       0.02350     0.03238    0.726 0.468009    
#      -0.10567     0.04112   -2.570 0.010181 *  
#      -0.03895     0.03236   -1.203 0.228800    
#       0.03417     0.04145    0.824 0.409803    
#      -0.01437     0.03261   -0.440 0.659601    
#       0.11593     0.03994    2.903 0.003701 ** 
#      -0.02682     0.03298   -0.813 0.416008    
#       0.01706     0.04224    0.404 0.686229    
#       0.05469     0.03279    1.668 0.095403 .  
#      -0.14014     0.04206   -3.332 0.000864 ***
# ---
# Signif. codes:  0 ¡®***¡¯ 0.001 ¡®**¡¯ 0.01 ¡®*¡¯ 0.05 ¡®.¡¯ 0.1 ¡® ¡¯ 1
# --- 
# Estimates in matrix form: 
# Constant term:  
# Estimates:  1.089476 0.4299534 
# MA coefficient matrix 
# MA( 1 )-matrix 
#        [,1]   [,2]
# [1,] 0.0263 -0.151
# [2,] 0.0235 -0.106
# MA( 2 )-matrix 
#         [,1]   [,2]
# [1,] -0.0865 0.1500
# [2,] -0.0389 0.0342
# MA( 3 )-matrix 
#         [,1]  [,2]
# [1,] -0.0528 0.119
# [2,] -0.0144 0.116
# MA( 4 )-matrix 
#         [,1]   [,2]
# [1,]  0.0178 0.0427
# [2,] -0.0268 0.0171
# MA( 5 )-matrix 
#        [,1]   [,2]
# [1,] 0.0403 -0.137
# [2,] 0.0547 -0.140
#   
# Residuals cov-matrix: 
#          [,1]     [,2]
# [1,] 47.74780 24.03471
# [2,] 24.03471 29.62013
# ---- 
# aic=  6.773557 
# bic=  6.881873 


#Estimation of a VMA(q) model using the exact likelihood method. Multivariate Gaussian likelihood function is used
VMAe(y, q = 5, include.mean = T)




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

# Coefficient(s):
#        Estimate  Std. Error  t value Pr(>|t|)    
#  [1,]  0.008197    0.015772    0.520  0.60323    
#  [2,]  0.013888    0.012798    1.085  0.27785    
#  [3,]  1.432590    0.155959    9.186  < 2e-16 ***
#  [4,] -0.207295    0.218667   -0.948  0.34313    
#  [5,] -0.496027    0.152753   -3.247  0.00117 ** 
#  [6,]  0.263464    0.215074    1.225  0.22058    
#  [7,] -0.132667    0.114584   -1.158  0.24694    
#  [8,]  1.368220    0.149780    9.135  < 2e-16 ***
#  [9,]  0.103714    0.112527    0.922  0.35669    
# [10,] -0.348084    0.147385   -2.362  0.01819 *  
# [11,] -0.284671    0.172108   -1.654  0.09812 .  
# [12,]  0.643123    0.236947    2.714  0.00664 ** 
# [13,]  0.247605    0.137115    1.806  0.07095 .  
# [14,] -0.059839    0.180700   -0.331  0.74053    
# ---
# Signif. codes:  0 ¡®***¡¯ 0.001 ¡®**¡¯ 0.01 ¡®*¡¯ 0.05 ¡®.¡¯ 0.1 ¡® ¡¯ 1
# --- 
# Estimates in matrix form: 
# Constant term:  
# Estimates:  0.008197498 0.01388816 
# AR coefficient matrix 
# AR( 1 )-matrix 
#        [,1]   [,2]
# [1,]  1.433 -0.207
# [2,] -0.133  1.368
# AR( 2 )-matrix 
#        [,1]   [,2]
# [1,] -0.496  0.263
# [2,]  0.104 -0.348
# MA coefficient matrix 
# MA( 1 )-matrix 
#        [,1]    [,2]
# [1,]  0.285 -0.6431
# [2,] -0.248  0.0598
#   
# Residuals cov-matrix: 
#             [,1]        [,2]
# [1,] 0.003575695 0.002501511
# [2,] 0.002501511 0.002205629
# ---- 
# aic=  -13.2787 
# bic=  -13.17254 

fit6=refVARMA(fit5)
#Refines a fitted VARMA model by setting insignificant estimates to zero

# Coefficient(s):
#      Estimate  Std. Error  t value Pr(>|t|)    
# Ph0  0.002459    0.002761    0.891 0.373151    
#      1.342995    0.057509   23.353  < 2e-16 ***
#     -0.344971    0.057545   -5.995 2.04e-09 ***
#      1.257516    0.056477   22.266  < 2e-16 ***
#     -0.260264    0.056338   -4.620 3.84e-06 ***
#     -0.090119    0.106945   -0.843 0.399414    
#      0.329649    0.086441    3.814 0.000137 ***
#      0.165413    0.042096    3.929 8.52e-05 ***
# ---
# Signif. codes:  0 ¡®***¡¯ 0.001 ¡®**¡¯ 0.01 ¡®*¡¯ 0.05 ¡®.¡¯ 0.1 ¡® ¡¯ 1
# --- 
# Estimates in matrix form: 
# Constant term:  
# Estimates:  0 0.00245852 
# AR coefficient matrix 
# AR( 1 )-matrix 
#      [,1] [,2]
# [1,] 1.34 0.00
# [2,] 0.00 1.26
# AR( 2 )-matrix 
#        [,1]  [,2]
# [1,] -0.345  0.00
# [2,]  0.000 -0.26
# MA coefficient matrix 
# MA( 1 )-matrix 
#         [,1]  [,2]
# [1,]  0.0901 -0.33
# [2,] -0.1654  0.00
#   
# Residuals cov-matrix: 
#             [,1]        [,2]
# [1,] 0.003659126 0.002538798
# [2,] 0.002538798 0.002231050
# ---- 
# aic=  -13.24636 
# bic=  -13.1857 



MTSdiag(fit6, gof = 10, adj = 0, level = T)
#level:	Logical switch for printing residual cross-correlation matrices

  
#Computes the forecasts of a VAR model, the associated standard errors of forecasts and the mean squared errors of forecasts
VARMApred(fit6, h=4)

