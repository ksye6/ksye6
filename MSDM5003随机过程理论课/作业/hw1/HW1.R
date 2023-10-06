library(ggplot2)

n=50000
x=rnorm(n,0,1)
y=rnorm(n,1,2^0.5)
z=rnorm(n,1,3^.5)
df1=data.frame(value=c(x,y,z),variables=rep(c("X","Y","Z"),each=n))

ggplot(df1)+geom_density(aes(x=value,fill=variables,color=variables),alpha=0.5,lwd=1.2,adjust=1.25)+
  ggtitle("Density Picture For X,Y And Z")+
  labs(caption="X~N(0,1) Y~N(1,2) Z~N(1,3)")+
  theme(plot.title = element_text(hjust = 0.5)) 

par(mfrow=c(1,2))
plot(density(z))
u=seq(-6,7,by=0.01)
plot(u,dnorm(u,1,3^.5),type="l")
