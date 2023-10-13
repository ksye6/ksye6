
sigma=1
k_values=seq(-10,10,by=0.01)

##expression:exp(-sigma^2*k^2/2)

mf=function(k){
  return(exp(-sigma^2*k^2/2))
}

plot(k_values,mf(k_values),type="l",ylab="The dependence of its amplitude on k",xlab="k",main="plot")