###################################################################################
#textbook page51

vw=read.table("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-3/chap3_R/m-ibm3dx2608.txt",header=T)[,3]
plot(vw,main=" ",ylab="",xlab="",type="l")
#acf(vw,20)

acf(vw,12,ylim=c(-0.2,0.2),main="")
pacf(vw,12,ylim=c(-0.2,0.2),main="")

Box.test(vw,lag=12,type="Ljung")


m3=arima(vw,order=c(3,0,0))
m3
# Call:
#   arima(x = vw, order = c(3, 0, 0))
# 
# Coefficients:
#   ar1      ar2      ar3  intercept
# 0.1158  -0.0187  -0.1042     0.0089
# s.e.  0.0315   0.0317   0.0317     0.0017
# 
# sigma^2 estimated as 0.002875:  log likelihood = 1500.86,  aic = -2991.73
(1-.1158+.0187+.1042)*mean(vw) # Compute the intercept phi(0).
#0.008967611
sqrt(m3$sigma2) # Compute standard error of residuals
#0.0536189
Box.test(m3$residuals,lag=12,type="Ljung")
# Box-Ljung test
# 
# data:  m3$residuals 
# X-squared = 16.352, df = 12, p-value = 0.1756
pv=1-pchisq(16.352,9) #Compute p-value using 9 degrees of freedom
pv
# [1] 0.05988496

#To fix the AR(2) coef to zero:

m3=arima(vw,order=c(3,0,0),fixed=c(NA,0,NA,0))

# The subcommand ’fixed’ is used to fix parameter values, where NA denotes estimation and 0 means fixing the
#parameter to 0. The ordering of the parameters can be found using m3$coef.
m3
# Call:
#   arima(x = vw, order = c(3, 0, 0), fixed = c(NA, 0, NA, NA))
# 
# Coefficients:
#   ar1  ar2      ar3  intercept
# 0.1136    0  -0.1063     0.0089
# s.e.  0.0313    0   0.0315     0.0017
# 
# sigma^2 estimated as 0.002876:  log likelihood = 1500.69,  aic = -2993.38
 (1-.1136+.1063)*.0089 # Compute phi(0)
#0.00883503
sqrt(m3$sigma2) # Compute residual standard error
#0.05362832
Box.test(m3$residuals,lag=12,type="Ljung")
# Box-Ljung test
# 
# data:  m3$residuals
# X-squared = 16.828, df = 12, p-value = 0.1562
pv=1-pchisq(17.853,10)
pv
#0.0782576


vw=read.table("C://Users//张铭韬//Desktop//学业//港科大//MSDM5053时间序列//课件//week3//chap3_R//m-ibm3dx2608.txt",header=T)[,3]
length(vw)#996
vw1=vw[1:984]

plot(vw1,main=" ",ylab="",xlab="",type="l")

acf(vw1,12,ylim=c(-0.2,0.2),main="")
pacf(vw1,12,ylim=c(-0.2,0.2),main="")

Box.test(vw1,lag=12,type="Ljung")

ar3=arima(vw1,order=c(3,0,0),fixed=c(NA,0,NA,0))
ar3

fore=predict(ar3,12)

fore

U=append(vw[984],fore$pred+1.96*fore$se)
L=append(vw[984],fore$pred-1.96*fore$se)
U
L

#p=vw[985:996]
plot(1:24,vw[973:996],ylim=c(-0.2,0.2),type="o",ylab="",xlab="",main="Forecasting")
lines(12:24,append(vw[984],fore$pred),type="o",col="red")
lines(12:24, U,type="l",col="blue")
lines(12:24, L,type="l",col="blue")
#points(temp.date[2:7],p,type="o")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))



