
vw=read.table("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-4/HSI-97-98.txt",header=T)[,2]
A=vw
length(A) #492

plot(vw,main=" ",ylab="",xlab="",type="l")

l_rtn=diff(log(vw))

length(l_rtn)#491
rtn=l_rtn[1:487]


plot(rtn,main=" ",ylab="",xlab="",type="l")


acf(rtn,12,ylim=c(-0.2,0.2),main="")
pacf(rtn,12,ylim=c(-0.2,0.2),main="")

Box.test(rtn,lag=12,type="Ljung")


m3=arima(rtn,order=c(3,0,0))
m3

Box.test(m3$residuals,lag=12,type="Ljung")

# Box-Ljung test
# 
# data:  m3$residuals 
# X-squared = 11.277, df = 12, p-value = 0.1756
pv=1-pchisq(11.277,9) #Compute p-value using 9 degrees of freedom
pv
# [1] 0.2571989

#To fix the AR(2) coef to zero:

m3=arima(rtn,order=c(3,0,0),fixed=c(NA,NA,NA,0))

# The subcommand ¡¯fixed¡¯ is used to fix parameter values, where NA denotes estimation and 0 means fixing the
#parameter to 0. The ordering of the parameters can be found using m3$coef.
m3
# 
sqrt(m3$sigma2) # Compute residual standard error
#0.05362832
Box.test(m3$residuals,lag=12,type="Ljung")
# Box-Ljung test
# 
# data:  m3$residuals
# X-squared = 11.271, df = 12, 
pv=1-pchisq(11.271,9)
pv



fore=predict(m3,4)

fore

U=append(rtn[487],fore$pred+1.96*fore$se)
L=append(rtn[487],fore$pred-1.96*fore$se)
U
L

#p=rtn[486:491]
plot(1:13,l_rtn[479:491],ylim=c(-0.2,0.2),type="o",ylab="",xlab="",main="Forecasting")
lines(9:13,append(rtn[487],fore$pred),type="o",col="red")
lines(9:13, U,type="l",col="blue")
lines(9:13, L,type="l",col="blue")
#points(temp.date[2:7],p,type="o")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))

################forecast for price P_{t}############################
##******************

est=arima(log(vw),c(3, 1, 0), seasonal = list(order = c(0,0,0), period = 1))
est


forecast=predict(est, n.ahead =4)
U1=forecast$pred +1.96 * forecast$se
L1=forecast$pred - 1.96 * forecast$se
U1
L1


E1=exp(forecast$pred+forecast$se*forecast$se/2)
E1

U2=append(vw[488],exp(U1))
L2=append(vw[488],exp(L1))
U2
L2


plot(1:13,vw[480:492],ylim=c(9000,13000),type="o",ylab="",xlab="",main="Forecasting")
lines(9:13,append(vw[488],E1),type="o",col="red")
lines(9:13, U2,type="l",col="blue")
lines(9:13, L2,type="l",col="blue")
#points(temp.date[2:7],p,type="o")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))

########################################

