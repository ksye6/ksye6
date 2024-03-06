
AP=c( .71,   .63,  .85,   .44,  .61,   .69,   .92,   .55 ,  .72 ,
      .77 ,  .92 ,  .6 ,  .83 ,  .8 ,  1 ,  .77,   .92 ,  1 ,  1.24, 
       1,  1.16,   1.3,  1.45,   1.25,   1.26,   1.38,   1.86,  1.56,   1.53, 
      1.59,   1.83,   1.86,   1.53,   2.07,   2.34,   2.25,   2.16,   2.43, 
      2.7,   2.25,   2.79,   3.42,   3.69,   3.6,   3.6,   4.32,   4.32, 
      4.05,   4.86,   5.04,   5.04,   4.41,   5.58,   5.85,   6.57,   5.31, 
       6.03,   6.39,   6.93,   5.85,   6.93,   7.74,   7.83,   6.12,   7.74, 
      8.91,   8.28,   6.84,   9.54,   10.26,   9.54,   8.73,   11.88,   12.06, 
     12.15,   8.91,   14.04,   12.96,   14.85,   9.99,   16.2,   14.67,   16.02, 
      11.61 )

length(AP)#84#

plot(AP, ylab="Passengers (1000s)", type="o", pch =20,col="blue")

x=ts(AP, start = c(1960, 1), end = c(1980, 4), frequency=4)
plot.ts(x,xlab="date",ylab="earnings",col="blue")
title(main="Quarterly earnings of Johnson and Johmson:from 1960 to 1980")

#Step 1
lgAP=log(AP)

#Step 2

w1=diff(lgAP)
plot(w1,xlab=" ",ylab=" ",main="w1 ",type="l",col="red",ylim=c(-0.5,0.5))

w2=diff(lgAP,4)
plot(w2,xlab=" ",ylab=" ",main="w2 ",type="l",col="red",ylim=c(-0.5,0.5))

w3=diff(diff(lgAP,4))
plot(w3,xlab=" ",ylab=" ",main="w3 ",type="l",col="red",ylim=c(-0.5,0.5))


#unitrootTest(lgAP,lags=1,type=c("c"),)

#unitrootTest(w3,lags=1,type=c("c"))

#Step 4 Choose p and q

acf(w3,15,main="",col="red",ylim=c(-0.5,1))
pacf(w3,15,main="",col="red",ylim=c(-0.5,1))

#step 5 Estimate the parameters
est=arima(w3, c(1, 0, 0), seasonal = list(order = c(1,0, 0),  period = 4))
est

#Step 6: Diagnostic checking.
Box.test(est$residuals,lag=12,type="Ljung")

pv=1-pchisq(12.068,10) #Compute p-value using 10 degrees of freedom
pv

#Step 7: Try other models
est=arima(w3, c(0, 0, 1), seasonal = list(order = c(0,0,1),include.mean = FALSE, period = 4))
est

Box.test(est$residuals,lag=12,type="Ljung")

pv=1-pchisq(10.16,10) #Compute p-value using 10 degrees of freedom
pv


#Step 8: Model Selection




est=arima(lgAP,c(0, 1, 1), seasonal = list(order = c(0,1,1), period = 4))
est


#Step 9:Forecasting
forecast=predict(est, n.ahead =12)
forecast$pred
forecast$se


U=forecast$pred +1.96 * forecast$se
L=forecast$pred - 1.96 * forecast$se



fore1=exp(forecast$pred+forecast$se*forecast$se/2)
U1=exp(U)
L1=exp(L)
fore1
U1
L1

U2=append(AP[84], U1)
L2=append(AP[84],L1)
fore2=append(AP,fore1)
fore3=append(AP[84],fore1)

fore2
U2
L2



plot(1:32,fore2[65:96],ylim=c(5,45),type="o", ylab="",xlab="",main="Forecasting")
lines(20:32,fore3,type="o",col="red")
lines(20:32, U2,type="l",col="blue")
lines(20:32, L2,type="l",col="blue")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))




 

