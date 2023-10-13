##MCMC测试

#我们的目标平稳分布是一个均值3，标准差2的正态分布，而选择的马尔可夫链状态转移矩阵Q(i,j)的条件转移概率
#是以i为均值,方差1的正态分布在位置j的值

# T1=5000
# pi=rep(0,T1)
# sigma=1
# t=1
# while(t<T1-1){
#   t=t+1
#   pi_star=rnorm(1,pi[t-1],sigma^0.5)
#   alpha=min(1,(dnorm(pi_star[1],3,2)/dnorm(pi[t-1],3,2)))
#   u=runif(1,0,1)
#   if (u<alpha){
#     pi[t]=pi_star[1]
#   }else{
#     pi[t]=pi[t-1]
#   }
# }
# 
# par(mfrow=c(2,2))
# plot(pi,dnorm(pi,3,2))
# plot(density(pi))
# hist(pi,breaks=50)
# qqnorm(pi)
# 

# Choose configurations (states) randomly, then weight them：

# a=sqrt(1/2/pi)

# a=sqrt(1/2/pi)
# T1=50000
# states=rep(0,T1)
# t=1
# sigma=1
# while(t<T1-1){
#   t=t+1
#   states_star=rnorm(1,states[t-1],sigma^0.5)
#   alpha=min(1,(exp(-0.5*(states_star[1]^2))/a)/(exp(-0.5*(states[t-1]^2))/a))
#   u=runif(1,0,1)
#   if (u<alpha){
#     states[t]=states_star[1]
#   }else{
#     states[t]=states[t-1]
#   }
# }

# Choose configurations first then weight them evenly:

a=sqrt(1/2/pi)
T1=100000
states=rep(0,T1)
t=1
sigma=1
vec=c(TRUE,FALSE)

while(t<T1-1){
  t=t+1
  states_star=rnorm(1,states[t-1],sigma^0.5)
  alpha=min(1,(exp(-0.5*(states_star[1]^2))/a)/(exp(-0.5*(states[t-1]^2))/a))
  u=sample(vec,size=1,replace=TRUE,prob=c(alpha,1-alpha))
  if (u==TRUE){
    states[t]=states_star[1]
  }else{
    states[t]=states[t-1]
  }
}

set.seed(1)

par(mfrow=c(2,2))

plot(density(states))
hist(states,breaks=50)
qqnorm(states)


f=function(x){
  return(exp(-0.5*(x^2))*sqrt(1/2/pi))
}

curve(f,from=-5,to=5,n=10000,xlab="x",ylab ="f(x)",main="Function Plot")

integrate(f,-100,100)

mean(states)
mean(states^2)
mean(states^3)
mean(states^4)
