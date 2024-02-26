

#simulation for unit root model

data=function(n,theta){
  m=1
  noise=rnorm(n+m)
  r=numeric(m+n)
  
  for (i in 2:(m+n))
    #    r[i]=theta[2]*r[i-1]+noise[i]
    
    r[i]=theta[1]+theta[2]*r[i-1]+noise[i]
  
  return(r[-(1:m)])
  
}

n=1000
#theta=c(0.5,0.2)
theta=c(0.0, 0.95)

sample=data(n,theta)

sample[1:100]


plot(sample,main=" ",ylab="",xlab="",type="l")

acf(sample,20,main=expression(paste("AR(1) model")),col="red")

pacf(sample,20,main=expression(paste("AR(1) model")),col="red")

adfTest(sample, lags = 1, type = c("c"), title = NULL,
        description = NULL)

#"nc" for a regression with no intercept (constant) nor time trend, and "c" for
#a regression with an intercept (constant) but no time trend, "ct" for a regression
#with an intercept (constant) and a time trend. The default is "c".









##########################################
#simulation for ARIMA
b=arima.sim(n = 1000, list(ar=0.8,order=c(1,1,0)))
plot(b,main=" ",ylab="",xlab="",type="l")

acf(b,20,main=expression(paste("ARIMA(1,1,0) models with ", phi[1]==0.8)),col="red")
pacf(b,20,main=expression(paste("ARIMA(1,1,0) models with ", phi[1]==0.8)),col="red")


adfTest(b, lags = 1, type = c("c"), title = NULL,
        description = NULL)

#"nc" for a regression with no intercept (constant) nor time trend, and "c" for
#a regression with an intercept (constant) but no time trend, "ct" for a regression
#with an intercept (constant) and a time trend. The default is "c".




db=diff(b)
plot(db,main=" ",ylab="",xlab="",type="l")

acf(db,20,main= "Differencing for ARIMA(1,1,0) models" ,col="red")
pacf(db,20,main= "Differencing for ARIMA(1,1,0) models",col="red")


b=arima.sim(n = 1000, list(ma=0.5,order=c(0,1,1)))
plot(b,main=" ",ylab="",xlab="",type="l")


acf(b,20,main=expression(paste("ARIMA(0,1,1) models with ", theta[1]==0.75)),col="red")
pacf(b,20,main=expression(paste("ARIMA(0,1,1) models with ", theta[1]==0.75)),col="red")

adfTest(sample, lags = 1, type = c("c"), title = NULL,
        description = NULL)

#"nc" for a regression with no intercept (constant) nor time trend, and "c" for
#a regression with an intercept (constant) but no time trend, "ct" for a regression
#with an intercept (constant) and a time trend. The default is "c".








db=diff(b)
plot(db,main=" ",ylab="",xlab="",type="l")

acf(db,20,main= "Differencing for ARIMA(0,1,1) models" ,col="red")
pacf(db,20,main= "Differencing for ARIMA(0,1,1) models",col="red")



b=arima.sim(n = 1000, list(ar=0.9,ma=0.5,order=c(1,1,1)))

plot(b,main=" ",ylab="",xlab="",type="l")


acf(b,20,main=expression(paste("ARIMA(1,1,1) models with ", phi[1]==0.9, " and ",theta[1]==0.5)),col="red")
pacf(b,20,main=expression(paste("ARIMA(1,1,1) models with ", phi[1]==0.9, " and ",theta[1]==0.5)),col="red")



adfTest(sample, lags = 1, type = c("c"), title = NULL,
        description = NULL)

#"nc" for a regression with no intercept (constant) nor time trend, and "c" for
#a regression with an intercept (constant) but no time trend, "ct" for a regression
#with an intercept (constant) and a time trend. The default is "c".






db=diff(b)
plot(db,main=" ",ylab="",xlab="",type="l")

acf(db,20,main= "Differencing for ARIMA(1,1,1) models" ,col="red")
pacf(db,20,main= "Differencing for ARIMA(1,1,1) models",col="red")



