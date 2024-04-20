gnp=read.table("C:/Users/sweet-xingxing/Desktop/time series paper/lidong/R for MTAR/USGNP.txt")[,2]
data=100*diff(log(gnp))
plot(gnp,type="l",col="blue",main="Quarterly US real GNP data ",xlab="",ylab="")# 1947¨C2009
plot(data,type="l",col="red",main="Growth rate ",xlab="",ylab="")


# #######################################################################
# ##
# ##                        Likelihood function
# ##
# #######################################################################
# RSS=function(theta,data,r1,r2){
#   #theta=(beta10,beta11,beta12,beta13,beta14,beta15,beta16;
#   #       beta20,beta21,beta22,beta23,beta24,beta25,beta26,beta27;                                  
    #       beta30,beta31,beta32,beta33,beta34,beta35,beta36,beta37,beta38,beta39,beta310)
#   #data is the real time series.
#   #r1 and r2 are the thresholds.
#   n=length(data)
#   A=matrix(numeric(10*(n-10)),(n-10),10)
#   for (i in 1:10){
#     A[,i]=data[(11-i):(n-i)]
#   }
#   Y=data[11:n]
#   D=matrix(numeric(26*(n-10)),(n-10),26)
#   I1=(data[5:(n-6)]<=r1)
#   I2=(data[5:(n-6)]>r1&data[5:(n-6)]<=r2)
#   I3=(data[5:(n-6)]>r2)
#   D[,1:7]=cbind(1,A[,1:6])*I1
#   D[,8:15]=cbind(1,A[,1:7])*I2
#   D[,16:26]=cbind(1,A)*I3
#   RSS=sum((Y-D%*%theta)^2)
#   return(RSS)
# }
# ###########################################################################
# ###                Estimation  when r1,r2 is known, d=6                 ###
# ###                Method1: using objective function
# ###                Computing time: 2hours
# ###########################################################################
# Estknown=function(data,r1,r2){
# n=length(data)
# initial=rep(0,26)
# fit=optim(initial,RSS, data=data,r1=r1,r2=r2,method="BFGS")
# hatpar=c(fit$par,r1,r2,fit$value)
# return(hatpar)
# }
###########################################################################
###                Estimation  when r1,r2 is known, d=6                 ###
###                Method2:  Calculate directly
###                Computing time: 16 seconds
###########################################################################
Estknown=function(data,r1,r2){
  n=length(data)
  A=matrix(numeric(10*(n-10)),(n-10),10)
  for (i in 1:10){
    A[,i]=data[(11-i):(n-i)]
  }
  Y=data[11:n]
  D=matrix(numeric(26*(n-10)),(n-10),26)
  I1=(data[5:(n-6)]<=r1)
  I2=(data[5:(n-6)]>r1&data[5:(n-6)]<=r2)
  I3=(data[5:(n-6)]>r2)
  D[,1:7]=cbind(1,A[,1:6])*I1
  D[,8:15]=cbind(1,A[,1:7])*I2
  D[,16:26]=cbind(1,A)*I3
  hatpar=solve(t(D)%*%D)%*%(t(D)%*%Y)
  RSS=sum((Y-D%*%hatpar)^2)
  return(c(hatpar,r1,r2,RSS)) 
} 


###########################################################################
###                Estimation  when r1,r2 is unknown                    ###
###########################################################################
Estunknown=function(data){
  n=length(data)
  alpha=0.1
  d=n*alpha
  r=sort(data)
  r1=r[d:(n-2*d)]
  r2=r[d:(n-d)]
  m=length(r1)
  estimate=matrix(numeric(29*m),m,29)
  for (i in 1:m){
    fit=t(sapply(r2[(i+d):(m+d)],Estknown,data=data,r1=r1[i]))
    estimate[i,]=fit[which.min(fit[,29]),]
  }
  result=estimate[which.min(estimate[,29]),]
  return(round(result[1:28],4))
}
date()
par=Estunknown(data)
par
date()
#############################################################################
# beta10  beta11  beta12  beta13  beta14  beta15  beta16;
# 0.6642  0.6351  0.2257 -0.2886  0.0875 -0.1899  0.0560
#
# beta20  beta21  beta22  beta23  beta24  beta25  beta26  beta27;          
# 0.1113 -0.0835 -0.2129  0.2898 -0.0139 -0.2432  0.6884  0.4081                          
#   
# beta30  beta31  beta32  beta33  beta34  beta35  beta36  beta37  beta38  beta39  beta310;
# 0.4959  0.1648  0.2463 -0.0844  0.3540  0.1530 -0.2924 -0.0732 -0.2003  0.2138  0.4330  
#
# r1=1.8762  r2=2.4266
#############################################################################
##
##                       Calculate the Variance of error
##
#############################################################################
sigma=function(data,par){
n=length(data)
A=matrix(numeric(10*(n-10)),(n-10),10)
for (i in 1:10){
  A[,i]=data[(11-i):(n-i)]
}
Y=data[11:n]
I1=(data[5:(n-6)]<=par[27])
I2=(data[5:(n-6)]>par[27]&data[5:(n-6)]<=par[28])
I3=(data[5:(n-6)]>par[28])
n1=sum(I1)
n2=sum(I2)
n3=sum(I3)
sigma1=sqrt(sum((Y-cbind(1,A[,1:6])%*%par[1:7])^2*I1)/n1)
sigma2=sqrt(sum((Y-cbind(1,A[,1:7])%*%par[8:15])^2*I2)/n2)
sigma3=sqrt(sum((Y-cbind(1,A)%*%par[16:26])^2*I3)/n3)
return(round(c(sigma1,sigma2,sigma3),4))
}
sigma(data,par)
#sigma1=0.7541, sigma2=0.8304, sigma3=0.8070
#############################################################################
##
##                            Asymptotic Variance
##
#############################################################################
ASD=function(data,par){
  s=sigma(data,par)
  n=length(data)
  A=matrix(numeric(10*(n-10)),(n-10),10)
  for (i in 1:10){
    A[,i]=data[(11-i):(n-i)]
  }
  AA=rbind(1,t(A))
  B1=matrix(numeric(7*7),7,7)
  B2=matrix(numeric(8*8),8,8)
  B3=matrix(numeric(11*11),11,11)
  for (i in 1:(n-10)){
    B1=B1+AA[1:7,i]%*%t(AA[1:7,i])*(A[i,6]<=par[27])
    B2=B2+AA[1:8,i]%*%t(AA[1:8,i])*(par[27]<A[i,6]&A[i,6]<=par[28])
    B3=B3+AA[1:11,i]%*%t(AA[1:11,i])*(A[i,6]>par[28])
  }
  v1=sqrt(diag(solve(B1/sum(A[,6]<=par[27]))))*s[1]
  v2=sqrt(diag(solve(B2/sum(par[27]<A[,6]&A[,6]<=par[28]))))*s[2]
  v3=sqrt(diag(solve(B3/sum(A[,6]>par[28]))))*s[3]
  asd=cbind(t(v1),t(v2),t(v3))/sqrt(n-10)  
  return(round(asd,4))
}
ASD(data,par)
#############################################################################
#    beta10  beta11  beta12  beta13  beta14  beta15  beta16;
#    0.6642  0.6351  0.2257 -0.2886  0.0875 -0.1899  0.0560
#ASD 0.1507  0.0590  0.0609  0.0634  0.0613  0.0582  0.0742
#
#    beta20  beta21  beta22  beta23  beta24  beta25  beta26  beta27;          
#    0.1113 -0.0835 -0.2129  0.2898 -0.0139 -0.2432  0.6884  0.4081                          
#ASD 0.8328  0.0531  0.0601  0.0644  0.0522  0.0588  0.3681  0.0581 
#
#    beta30  beta31  beta32  beta33  beta34  beta35  beta36  beta37  beta38  beta39  beta310;
#    0.4959  0.1648  0.2463 -0.0844  0.3540  0.1530 -0.2924 -0.0732 -0.2003  0.2138  0.4330 
#ASD 0.2804  0.0559  0.0539  0.0492  0.0735  0.0608  0.0802  0.0503  0.0481  0.0553  0.0444

#############################################################################
##
##                            Diagnostic of the residuals
##
#############################################################################
n=length(data)
A=matrix(numeric(10*(n-10)),(n-10),10)
for (i in 1:10){
  A[,i]=data[(11-i):(n-i)]
}
Y=data[11:n]
D=matrix(numeric(26*(n-10)),(n-10),26)
I1=(data[5:(n-6)]<=par[27])
I2=(data[5:(n-6)]>par[27]&data[5:(n-6)]<=par[28])
I3=(data[5:(n-6)]>par[28])
D[,1:7]=cbind(1,A[,1:6])*I1
D[,8:15]=cbind(1,A[,1:7])*I2
D[,16:26]=cbind(1,A)*I3
resd=Y-D%*%par[1:26]

Box.test(resd,lag=6,type ="Ljung-Box")  ### Box test is used
Box.test(resd^2,lag=6,type ="Ljung-Box") ### Li-MaK test is used here


