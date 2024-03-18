
#simulation for GARCH(1,1) model
library(fGarch) # Load the package

data=function(n,alpha){
  m=2000
  noise=rnorm(n+m)
  a=numeric(m+n)
  h=numeric(m+n)
  r=numeric(m+n)
    
  for (i in 2:(m+n))
    h[i]=alpha[1]+(alpha[2]*noise[i-1]*noise[i-1]+alpha[3])*h[i-1]
  
  for (i in 2:(m+n))
      a[i]=sqrt(h[i])*noise[i]

  for (i in 2:(m+n))
    r[i]=alpha[4]*r[i-1]+a[i]
  
    
  return(h[-(1:m)])
}



n=1000

alpha=c(0.1,0.15,0.8,0.0)

sample=data(n, alpha)

sample[1:100]

plot(sample,main=" ",ylab="",xlab="",type="l")


mean(sample)
var(sample)
sqrt(var(sample)) # Standard deviation
skewness(sample)
kurtosis(sample)





acf(sample,10,main=expression(paste("GARCH(1,1)")),col="red")

Box.test(sample,lag=12,type="Ljung")

plot(sample*sample,main=" ",ylab="",xlab="",type="l")

acf(sample*sample,10,main=expression(paste("GARCH(1,1)")),col="red")

Box.test(sample*sample,lag=12,type="Ljung")


m5=garchFit(sample~arma(1,0)+garch(1,1),data=sample,trace=F)
summary(m5)


stresi=residuals(m5,standardize=T)
plot(stresi,type="l")
Box.test(stresi,10,type="Ljung")
Box.test(stresi^2,10,type="Ljung")
predict(m5,5)


