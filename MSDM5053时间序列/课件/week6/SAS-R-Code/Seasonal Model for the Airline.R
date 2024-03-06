
AP=c(112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
204, 188, 235, 227, 234, 264, 302, 293, 259 ,229, 203, 229,
242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432)
length(AP)#144
#plot(AP, ylab="Passengers (1000s)", type="o", pch =20,col="blue")

x=ts(AP, start = c(1949, 1), end = c(1960, 12), frequency=12)
plot.ts(x,xlab="date",ylab="Passengers",col="blue")
title(main="International Airline Passengers from 1949 to 1960")

#Step 1
lgAP=log(AP)

#Step 2

w1=diff(lgAP)
plot(w1,xlab=" ",ylab=" ",main="w1 ",type="l",col="red",ylim=c(-0.35,0.35))

w2=diff(lgAP,12)
plot(w2,xlab=" ",ylab=" ",main="w2 ",type="l",col="red",ylim=c(-0.1,0.35))

w3=diff(diff(lgAP,12))
plot(w3,xlab=" ",ylab=" ",main="w3 ",type="l",col="red",ylim=c(-0.35,0.35))


#unitrootTest(lgAP,lags=1,type=c("c"),)

#unitrootTest(w3,lags=1,type=c("c"))

#Step 4 Choose p and q

acf(w3,15,main="",col="red",ylim=c(-0.5,1))
pacf(w3,15,main="",col="red",ylim=c(-0.5,1))

#step 5 Estimate the parameters
est=arima(w3, c(1, 0, 0), seasonal = list(order = c(1,0, 0),  period = 12))
est

#Step 6: Diagnostic checking.
Box.test(est$residuals,lag=12,type="Ljung")

pv=1-pchisq(13.89,10) #Compute p-value using 10 degrees of freedom
pv

#Step 7: Try other models
est=arima(w3, c(0, 0, 1), seasonal = list(order = c(0,0,1),include.mean = FALSE, period = 12))
est

#Step 8: Model Selection




est=arima(lgAP,c(0, 1, 1), seasonal = list(order = c(0,1,1), period = 12))
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

U2=append(AP[144], U1)
L2=append(AP[144],L1)
fore2=append(AP,fore1)
fore3=append(AP[144],fore1)

fore2
U2
L2



plot(1:32,fore2[125:156],ylim=c(200,800),type="o", ylab="",xlab="",main="Forecasting")
lines(20:32,fore3,type="o",col="red")
lines(20:32, U2,type="l",col="blue")
lines(20:32, L2,type="l",col="blue")
legend(x="topleft",c("True returns","prediction"),lty=c(1,1),pch=c(1,1),col=c("black","red"))




 

