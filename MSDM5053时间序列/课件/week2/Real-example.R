
setwd("C:/ling/teaching/teaching/MSBD5006MSDM5053/Lecture-2/data")
# set the working directory

da=read.csv("N225-84-09.csv",header=T) 
#header=T means 1st row of the data file contains variable names. The default is header=F, i.e., no names.

A=da[,5]
#See the first column of the data

ndaily=length(A)
A[1:100]
ndaily

#Compute the simple return
s_rtn=(A[2:ndaily]/A[1:(ndaily-1)]-1)*100



#Compute the log return 
l_rtn=diff(log(A))*100 
par(mfrow=c(2,2))
#s_rtn=exp(l_rtn)-1
plot(A,main="Daily closing price",ylab="",xlab="",type="l")
plot(s_rtn,main="Daily simple return",ylab="",xlab="",type="l")
plot(l_rtn,main="Daily log return",ylab="",xlab="",type="l")




hist(l_rtn, 30, fre=F, col="blue",xlim=c(-6,6),ylim=c(0,0.5),
     main="",ylab="Histogram",xlab="")

x=seq(-6,6,0.01)
lines(x,dnorm(x,mean(l_rtn),sd(l_rtn)),lty=2,lwd=1.2, col="red",xlim=c(-6,6),ylim=c(0,0.5))
legend("topright",c("log return", "Normal"),lty=c(1,2),lwd=1.2, col=c("red","green3"),cex=0.6)




# Compute the summary statistics
library(fBasics) # Load the package fBasics.
#basicStats(s_rtn) 

#Alternatively, one can use individual commands as follows:
mean(s_rtn)
var(s_rtn)
sqrt(var(s_rtn)) # Standard deviation
skewness(s_rtn)
kurtosis(s_rtn)
quantile(s_rtn, prob=c(0.01, 0.99), type = 1)


mean(l_rtn)
var(l_rtn)
sqrt(var(l_rtn)) # Standard deviation
skewness(l_rtn)
kurtosis(l_rtn)
quantile(l_rtn, prob=c(0.01, 0.99), type = 1)

#Simple tests
n=length(l_rtn)
s1=skewness(l_rtn)
t1=s1/sqrt(6/n) #Compute test statistic
t1

pv=2*pnorm(t1) #Compute p-value.
pv

pv1=2*(1-pnorm(t1)) #Compute p-value.
pv1

#The result shows that the hypothesis of zero expected return can be rejected at the 5%  level.

#Normality test
normalTest(l_rtn,method="jb") 
#The result shows the normality for simple return is rejected
#perform Kolmogorov-Smirnov test
library("dgof")
ks.test(l_rtn, "pnorm")


#Test mean being zero.
t.test(l_rtn) 


# test for ACF=0
Box.test(l_rtn,lag=1,type="Ljung")
Box.test(l_rtn,lag=6,type="Ljung")
Box.test(l_rtn,lag=12,type="Ljung")



