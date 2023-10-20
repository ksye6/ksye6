#3

DRlist=c(0.2,0.5,1,3)
tlist=seq(0,8,0.02)

integration=function(DR,t,T=10000){
  set.seed(1)
  samples=rnorm(n=T,0,1)
  integrand=cos(sqrt(DR*2)*sqrt(t)*samples)
  integral=sum(integrand)/T
  return(integral)
}

par(mfrow=c(2,2))

T=100000
result=c()

for (DR in DRlist){
  for (i in 1:length(tlist)){
    result[i]=integration(DR,tlist[i],T)
  }
  
  plot(tlist,result,type="p",col="black",main=paste("DR=",DR))
  curve(exp(-DR*x),from=0,to=4,add=TRUE,col="red",lwd=1.5)
  legend("topright", legend = c("simulation", "exp(-DR*t)"), col = c("black", "red"), lty = 1)
}

par(mfrow=c(1,1))

#求Tau去对比Tau和DR的关系
model=function(t, Tau) exp(-t / Tau)
init_params=list(Tau=1)
DRlist2=c(0.1,0.3,0.5,0.7,0.9,1,1.2,1.3,1.5,2,3,5,8)
Taulist=c()

for (DR in DRlist2){
  for (i in 1:length(tlist)){
    result[i]=integration(DR,tlist[i],T)
  }
  
  fit=nls(result ~ model(tlist,Tau), start =init_params)
  Tau=coef(fit)["Tau"]
  Taulist=c(Taulist,Tau)
}

plot(DRlist2,Taulist,type="b",col="black",lwd=2,main="DR-Tau Relationship")
curve(1/x,from=0,to=10,add=TRUE,col="red",lwd=1)
legend("topright", legend = c("DR-Tau-simulation", "1/x"), col = c("black", "red"), lty = 1)


#5
v=2
tau=2






