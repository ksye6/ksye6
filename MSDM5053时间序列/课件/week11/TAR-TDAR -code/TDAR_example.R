weeklydata=read.csv("C:/Users/sweet-xingxing/Desktop/time series paper/lidong/QMLE of TDAR/weeklydata.csv")$Close
data=100*diff(log(weeklydata))
plot(weeklydata,type="l",col="blue",main="weekly closing prices of HSI ",xlab="",ylab="")# January 2000 - December 2007
plot(data,type="l",col="red",main="Log-return of weekly closing prices of HSI ",xlab="",ylab="")

#######################################################################
##
##                        Likelihood function
##
#######################################################################
likfun=function(theta,data,r){
  #theta=(phi10,phi11,phi12,alpha10,alpha11,alpha12,alpha13; phi20,phi21,phi22,alpha20,alpha22,alpha23)
  #data is the real time series.
  #r is the threshold.
  n=length(data)
  y1=data[1:(n-3)]
  y2=data[2:(n-2)]
  y3=data[3:(n-1)]
  y4=data[4:n]
  sigma1=theta[4]+theta[5]*y3^2+theta[6]*y2^2+theta[7]*y1^2
  sigma2=theta[11]+theta[12]*y2^2+theta[13]*y1^2
  s1=(log(sigma1)+(y4-theta[1]-theta[2]*y3-theta[3]*y2)^2/sigma1)*(y3<=r)
  s2=(log(sigma2)+(y4-theta[8]-theta[9]*y3-theta[10]*y2)^2/sigma2)*(y3>r)
  loglik=0.5*sum(s1+s2)
  return(loglik)
}

##########################################################################
##
##                               Estimation 
##
##########################################################################
est=function(data){
  n=length(data)
  d=0.2*n
  r=sort(data)[(d+1):(n-d)]
  initial=c(rep(0,3),rep(0.1,4),rep(0,3),rep(0.1,3))
  p=length(initial)
  m=length(r)
  A=matrix(rep(0,(p+2)*m),m,(p+2))
  for (i in 1:m){
    fit=optim(initial,likfun,data=data,r=r[i],method="L-BFGS-B",gr=NULL,
              lower=c(rep(-5,3),rep(0.001,4),rep(-5,3),rep(0.001,3)), upper=rep(20,13))
    A[i,]=c(t(fit$par),r[i],fit$value)
  }
  est=round(A[which.min(A[,(p+2)]),],4)
  return(est)
}
result=est(data)
result
#############################################################################
#  phi10    phi11   phi12    alpha10  alpha11  alpha12  alpha13;    threshold
#-0.2442  -0.1572   0.2696   4.3458   0.5242   0.1745   0.0975       0.0234
#
#  phi20    phi21   phi22    alpha20           alpha22  alpha23 
#-0.0951   0.0933  -0.0726   4.0206            0.0764   0.1389 
#     
#log-likelihood:613.1637
#
#############################################################################
##
##                            Asymptotic Variance
##
#############################################################################
ASD=function(data,par){
  #par is the estimated parameter.
  n=length(data)
  y1=data[1:(n-3)]
  y2=data[2:(n-2)]
  y3=data[3:(n-1)]
  y4=data[4:n]
  
  q=length(par)
  sigma1=par[4]+par[5]*y3^2+par[6]*y2^2+par[7]*y1^2
  sigma2=par[11]+par[12]*y2^2+par[13]*y1^2
  error=(y4-par[1]-par[2]*y3-par[3]*y2)/sqrt(sigma1)*(y3<=par[q])+(y4-par[8]-par[9]*y3-par[10]*y2)/sqrt(sigma2)*(y3>par[q])
  
  A1=rbind(1,y3,y2)
  B1=rbind(1,y3^2,y2^2,y1^2)
  
  A2=rbind(1,y3,y2)
  B2=rbind(1,y2^2,y1^2)
  
  a1=nrow(A1)
  b1=nrow(B1)
  a2=nrow(A2)
  b2=nrow(B2)
  p1=b1+a1
  p2=b2+a2
  
  sigma=matrix(numeric((p1+p2)^2),(p1+p2),(p1+p2))
  omega=matrix(numeric((p1+p2)^2),(p1+p2),(p1+p2))
  
  delta11=matrix(numeric(a1^2),a1,a1)
  delta12=matrix(numeric(b1^2),b1,b1)
  delta13=matrix(numeric(a1*b1),a1,b1)
  
  delta21=matrix(numeric(a2^2),a2,a2)
  delta22=matrix(numeric(b2^2),b2,b2)
  delta23=matrix(numeric(a2*b2),a2,b2)
  
  
  for (i in 1:(n-4)){ 
    delta11=delta11+A1[,i]%*%t(A1[,i])*(y3[i]<=par[q])/(par[4]+par[5]*y3[i]^2+par[6]*y2[i]^2+par[7]*y1[i]^2)
    delta12=delta12+B1[,i]%*%t(B1[,i])*(y3[i]<=par[q])/(par[4]+par[5]*y3[i]^2+par[6]*y2[i]^2+par[7]*y1[i]^2)^2
    delta13=delta13+A1[,i]%*%t(B1[,i])*(y3[i]<=par[q])/(par[4]+par[5]*y3[i]^2+par[6]*y2[i]^2+par[7]*y1[i]^2)^(3/2)
    
    delta21=delta21+A2[,i]%*%t(A2[,i])*(y3[i]>par[q])/(par[11]+par[12]*y2[i]^2+par[13]*y1[i]^2)
    delta22=delta22+B2[,i]%*%t(B2[,i])*(y3[i]>par[q])/(par[11]+par[12]*y2[i]^2+par[13]*y1[i]^2)^2
    delta23=delta23+A2[,i]%*%t(B2[,i])*(y3[i]>par[q])/(par[11]+par[12]*y2[i]^2+par[13]*y1[i]^2)^(3/2)
    
  }
  
  k3=mean(error^3)
  k4=mean(error^4)

  omega[1:a1,1:a1]=delta11
  omega[(a1+1):p1,(a1+1):p1]=0.5*delta12
  omega[(p1+1):(p1+a2),(p1+1):(p1+a2)]=delta21
  omega[(p1+a2+1):(p1+p2),(p1+a2+1):(p1+p2)]=0.5*delta22
  
  sigma[1:a1,1:a1]=delta11
  sigma[1:a1,(a1+1):p1]=0.5*k3*delta13
  sigma[(a1+1):p1,1:a1]=0.5*k3*t(delta13)
  sigma[(a1+1):p1,(a1+1):p1]=1/4*(k4-1)*delta12
  
  sigma[(p1+1):(p1+a2),(p1+1):(p1+a2)]=delta21
  sigma[(p1+1):(p1+a2),(p1+a2+1):(p1+p2)]=0.5*k3*delta23
  sigma[(p1+a2+1):(p1+p2),(p1+1):(p1+a2)]=0.5*k3*t(delta23)
  sigma[(p1+a2+1):(p1+p2),(p1+a2+1):(p1+p2)]=1/4*(k4-1)*delta22
  
  v=sqrt(diag(solve(omega/(n-4))%*%(sigma/(n-4))%*%solve(omega/(n-4))))
  asd=v/sqrt(n-4)
  
  return(asd)
}
par=result[1:14]
asd=round(ASD(data,par),4)
asd
###############################################################################################################################
#
#      phi10    phi11    phi12    alpha10  alpha11  alpha12  alpha13;   phi20    phi21    phi22    alpha20   alpha22  alpha23   
#     -0.2442  -0.1572   0.2696   4.3458   0.5242   0.1745   0.0975    -0.0951   0.0933  -0.0726   4.0206    0.0764   0.1389  
#ASD   0.3122   0.1486   0.0874   1.0944   0.1659   0.1237   0.0830     0.2544   0.0931   0.0618   0.6366    0.0584   0.0793
#  
###############################################################################################################################



#############################################################################
##
##                            Diagnostic of the residuals
##
#############################################################################
n=length(data)
y1=data[1:(n-3)]
y2=data[2:(n-2)]
y3=data[3:(n-1)]
y4=data[4:n]
q=length(par)
sigma1=par[4]+par[5]*y3^2+par[6]*y2^2+par[7]*y1^2
sigma2=par[11]+par[12]*y2^2+par[13]*y1^2
resd=(y4-par[1]-par[2]*y3-par[3]*y2)/sqrt(sigma1)*(y3<=par[q])+(y4-par[8]-par[9]*y3-par[10]*y2)/sqrt(sigma2)*(y3>par[q])

Box.test(resd,lag=6,type ="Ljung-Box")  ### Box test is used
Box.test(resd,lag=12,type ="Ljung-Box")  

Box.test(resd^2,lag=6,type ="Ljung-Box") ### Li-MaK test is used here
Box.test(resd^2,lag=12,type ="Ljung-Box")


