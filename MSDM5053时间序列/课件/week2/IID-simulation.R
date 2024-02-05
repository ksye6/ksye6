
#simulation 

data=function(n){
  #noise=2*rnorm(n)
  noise=rt(n,4)
  }

n=1000

sample=data(n)

sample[1:100]

plot(sample,main=" ",ylab="",xlab="",type="l")


hist(sample, 30, fre=F, col="blue",xlim=c(-6,6),ylim=c(0,0.4),
     main="",ylab="Histogram",xlab="")

x=seq(-6,6,0.1)
lines(x,dnorm(x,mean(sample),sd(sample)),lty=2,lwd=1.2, col="red",xlim=c(-6,6),ylim=c(0,0.4))
legend("topright",c("log return", "Normal"),lty=c(1,2),lwd=1.2, col=c("red","green3"),cex=0.6)




library(fBasics) # Load the package fBasics.
basicStats(sample) 

#Alternatively, one can use individual commands as follows:
mean(sample)
var(sample)
sqrt(var(sample)) # Standard deviation
skewness(sample)
kurtosis(sample)



t1=skewness(sample)/sqrt(6/n) #Compute test statistic
t1

pv1=2*pnorm(t1) #Compute p-value.
pv1
pv=2*(1-pnorm(t1)) #Compute p-value.
pv


#Test mean being zero.
t.test(sample) 
#The result shows that the hypothesis of zero expected return can be rejected at the 5%  level.

#Normality test
normalTest(sample,method="jb") 
#The result shows the normality for simple return is rejected




