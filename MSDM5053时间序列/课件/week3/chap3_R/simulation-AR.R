
#simulation for AR(1) model

data=function(n,theta){
  m=200
  noise=rnorm(n+m)
  r=numeric(m+n)
  
  for (i in 2:(m+n))
    r[i]=theta[2]*r[i-1]+noise[i]
    
    #r[i]=theta[1]+theta[2]*r[i-1]+noise[i]
  
  return(r[-(1:m)])
  
}

n=1000
#theta=c(0.5,0.2)
theta=c(0.0, 0.5)

sample=data(n,theta)

sample[1:100]


plot(sample,main=" ",ylab="",xlab="",type="l")

acf(sample,10,main=expression(paste("AR(1) with ", phi[1]==0.8)),col="red")
#pacf(b,20,main=expression(paste("AR(1) models with ", phi[2]==0.8)),col="red")


Box.test(sample,lag=6,type="Ljung")
Box.test(sample,lag=12,type="Ljung")
Box.test(sample,lag=24,type="Ljung")


########################################### AR(2) model

b=arima.sim(n = 1000, list(ar=c(1,-0.9)))
plot(b,main=" ",ylab="",xlab="",type="l")
acf(b,20,main=expression(paste("AR(2) with ", phi[1]==1," and ",phi[2]==-0.9)),col="red")
#pacf(b,20,main=expression(paste("AR(2) models with ", phi[1]==1," and ",phi[2]==-0.9)),col="red")

            
Box.test(b,lag=6,type="Ljung")
Box.test(b,lag=12,type="Ljung")
Box.test(b,lag=24,type="Ljung")



####stationary region for AR(2) model
phi1 <- seq(from = -2.5, to = 2.5, length = 51) 
plot(phi1,1+phi1,lty="dashed",type="l",xlab="",ylab="",cex.axis=.8,ylim=c(-1.2,1.2))
abline(a = -1, b = 0, lty="dashed")
abline(a = 1, b = -1, lty="dashed")
title(ylab=expression(phi[2]),xlab=expression(phi[1]),cex.lab=.8)
polygon(x = phi1[6:46], y = 1-abs(phi1[6:46]), col="gray")
lines(phi1,-phi1^2/4)
text(0,-.5,expression(phi[2]+phi[1]^2/4<0),cex=.7)
text(1.2,.5,expression(phi[2]+phi[1]>1),cex=.7)
text(-1.75,.5,expression(phi[2]-phi[1]>1),cex=.7)

############### AR(3)model #######################

b=arima.sim(n = 1000, list(ar=c(0.7,-0.2,-0.3)))

plot(b,main=" ",ylab="",xlab="",type="l")

acf(b,20,main=expression(paste("AR(3) with ", phi=="(0.7,-0.2,-0.3)" )),col="red")
#pacf(b,20,main=expression(paste("AR(3) with ", phi=="(0.7,-0.2,-0.3)" )),col="red")





###################################################################################


vw=arima.sim(n = 1000, list(ar=c(1,-0.9)))

length(vw)#1000
vw1=vw[1:990]

plot(vw1,main=" ",ylab="",xlab="",type="l")
acf(vw1,20,main=expression(paste("AR(2) with ", phi[1]==1," and ",phi[2]==-0.9)),col="red")
#pacf(b,20,main=expression(paste("AR(2) models with ", phi[1]==1," and ",phi[2]==-0.9)),col="red")


Box.test(vw1,lag=6,type="Ljung")
Box.test(vw1,lag=12,type="Ljung")
Box.test(vw1,lag=24,type="Ljung")



ar3=arima(vw1,order=c(2,0,0),fixed=c(NA,NA,0))
ar3

Box.test(ar3$residuals,lag=12,type="Ljung")
# Box-Ljung test
 
# data:  ar3$residuals
# X-squared = 3.0561, df = 12
pv=1-pchisq(3.0561,10)
pv


fore=predict(ar3,10)

fore

U=append(vw[990],fore$pred+1.96*fore$se)
L=append(vw[990],fore$pred-1.96*fore$se)
U
L


plot(1:25,vw[976:1000],ylim=c(-10,10),type="o",ylab="",xlab="",main="Forecasting")
lines(15:25,append(vw[990],fore$pred),type="o",col="red")
lines(15:25, U,type="l",col="blue")
lines(15:25, L,type="l",col="blue")
#points(temp.date[2:7],p,type="o")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))



