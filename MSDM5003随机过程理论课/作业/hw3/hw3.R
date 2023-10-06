s_0=curve(sin(2*pi*x),from = -3, to = 3)
p=curve(sin(20*pi*x),from = -3, to = 3)
s=curve(sin(2*pi*x)+sin(20*pi*x),from = -3, to = 3)


#(1):0<=x<=6;-3<=y<=3-x
##direct integration
ccs_0=function(y,x){
    return(sin(2*pi*y)*sin(2*pi*(y+x)))
}

x_values=seq(0,5.999,by=0.001)
results=list()

for (x in x_values) {
  result=integrate(ccs_0,-3,3-x,x=x)
  results[[as.character(x)]]=result$value
}

results_df=data.frame(x=x_values,integral=unlist(results))

plot(results_df$x,results_df$integral,type="l",ylab="integral",xlab="x",main="direct integration")


##expression:0.5*(cos(2*pi*x)*(6-x)-1/(4*pi)*(sin(12*pi-2*pi*x)-sin(2*pi*x-12*pi)))
s_00=function(x){
  return(0.5*(cos(2*pi*x)*(6-x)-1/(4*pi)*(sin(12*pi-2*pi*x)-sin(2*pi*x-12*pi))))
}
plot(x_values,s_00(x_values),type="l",ylab="integral",xlab="x",main="expression plot")

#(2)
delta=0.2  #delta决定整体平滑度（模糊度
k=qnorm(0.51,mean=0,sd=sqrt(delta))  #假设分位数>0.5，越大则内外幅度比越大，从0趋向于正无穷
# -k=<x-y<=k
# y:(max(-3,x-k),min(3,x+k))

conver_sr=function(y,x){
  return((sin(2*pi*y)+sin(20*pi*y))*(dnorm(x-y,mean=0,sd=sqrt(delta))))
}

x_values=seq(-10,10,by=0.05)
results=list()

for (x in x_values) {
  result=integrate(conver_sr,max(-3,x-k),min(3,x+k),x=x)
  results[[as.character(x)]]=result$value
}

results_df=data.frame(x=x_values,integral=unlist(results))

plot(results_df$x,results_df$integral,type="l",ylab="integral",xlab="x",main="direct integration",ylim=c(-0.5,0.5))

s_=function(y){
  return(sin(2*pi*y)+sin(20*pi*y))
}
plot(x_values,s_(x_values),type="l",ylab="integral",xlab="x",main="expression plot")

