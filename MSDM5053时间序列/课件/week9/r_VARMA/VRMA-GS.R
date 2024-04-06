
###Example 8.4(page 407)
rm(list = ls())
## remove (almost) everything in the working environment.

setwd("C:/ling/teaching/Shanxi-2021/r_VARMA")  #set my working directory
###########################################################################################


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

fit7=VARMA(lgy, p=4, include.mean = T)

MTSdiag(fit7, gof = 10, adj = 0, level = T)


fit8=refVARMA(fit7)


MTSdiag(fit8, gof = 10, adj = 0, level = T)


#Computes the forecasts of a VAR model, the associated standard errors of forecasts and the mean squared errors of forecasts
VARMApred(fit7, h=4)

