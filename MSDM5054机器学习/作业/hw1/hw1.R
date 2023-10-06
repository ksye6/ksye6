###1
#(1)
csv1=read.csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw1//Life Expectancy Data.csv")
df1=csv1[,-c(1,2)]
df1=na.omit(df1)
mod1=lm(Life.expectancy~.,data=df1)
summary(mod1)
#Adult.Mortality, infant.deaths, under.five.deaths, BMI, HIV.AIDS, Income.composition.of.resources, Schooling may be the most
#important variables; next are: Status, Alcohol, Diphtheria and percentage.expenditure.

#总体检验
par(mfrow=c(2,2))
plot(mod1, which=1:4, col="blue")
#独立性
library(DescTools)
DurbinWatsonTest(mod1)
#残差正态性
lmres=residuals(mod1)
shapiro.test(lmres)
par(mfrow=c(1,1))
qqnorm(lmres)
#方差齐性
library(zoo)
library(lmtest)
bptest(mod1)

#(2)
confint(mod1,level=0.95)
#Adult.Mortality: [-1.849399e-02,-1.476948e-02]
#       HIV.AIDS: [-4.719608e-01,-4.019673e-01]

#These 2 variables both pass the t-test(α=.001) so I'm confident they have negative impact on the life expectancy.

#(3)
confint(mod1,level=0.97)
#Schooling:[7.374800e-01,9.955264e-01]----→The larger the number of years of Schooling is, the longer the Life Expectancy is.
#  Alcohol:[-1.634257e-01 -1.936430e-02]--→The more alcohol consumption recorded per capita (15+) is, the shorter the Life Expectancy is.

#(4)
#Adult.Mortality:< 2e-16
#infant.deaths：< 2e-16
#under.five.deaths：< 2e-16
#BMI：2.15e-08
#HIV.AIDS：< 2e-16
#Income.composition.of.resources：< 2e-16
#Schooling：< 2e-16
mod2=lm(Life.expectancy~Adult.Mortality+infant.deaths+under.five.deaths+BMI+HIV.AIDS+Income.composition.of.resources+Schooling,data=df1)
summary(mod2)

#(5)
new_obs=data.frame(Adult.Mortality=125,infant.deaths=94,under.five.deaths=2,BMI=55,HIV.AIDS=0.5,Income.composition.of.resources=0.9,Schooling=18)
# create the new observation
predict(mod2,newdata=new_obs,interval="confidence",level=0.99) 

#(6)
AIC(mod1)
AIC(mod2)
#The AIC of mod1(full model) is smaller than mod2(smaller model).



###2
#(1)
train=read.csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw1//BreastCancer_train.csv")
train=train[,-1]
test=read.csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw1//BreastCancer_test.csv")
test=test[,-1]

train$type[which(train$Class== "benign")]=0 # benign编号为0
train$type[which(train$Class== "malignant")]=1 # malignant编号为1
test$type[which(test$Class== "benign")]=0 # benign编号为0
test$type[which(test$Class== "malignant")]=1 # malignant编号为1

train=train[,-10]
train=na.omit(train)
test=test[,-10]
test=na.omit(test)

train$type=factor(train$type)
test$type=factor(test$type)

glmod1=glm(type~., data=train,family=binomial)
summary(glmod1)
#Marg.adhesion, Bare.nuclei and Bl.cromatin are significant variables.
pred1=predict(glmod1,test,interval="prediction",type="response")
pred1=ifelse(pred1>0.5,1,0)
library(ROCR)
p1=prediction(pred1,test$type) #预测值和真实值 
perf1=performance(p1,"tpr","fpr")
plot(perf1,colorize=TRUE,lwd=4)
#AUC:0.943
performance(p1, "auc")@y.values[[1]]

#Another method:
library(pROC)
modelroc1=roc(test$type,pred1) #真实值和预测值
plot(modelroc1, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)


#(2)
glmod2=glm(type~Cl.thickness+Cell.shape+Marg.adhesion+Bare.nuclei+Bl.cromatin, data=train,family=binomial)
summary(glmod2)
pred2=predict(glmod2,test,interval="prediction",type="response")
pred2=ifelse(pred2>0.5,1,0)

p2=prediction(pred2,test$type) #预测值和真实值 
perf2=performance(p2,"tpr","fpr")
plot(perf2,colorize=TRUE,lwd=4)
#AUC:0.953
performance(p2, "auc")@y.values[[1]]

modelroc2=roc(test$type,pred2) #真实值和预测值
plot(modelroc2, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

#The AUC of glmod2 is larger than glmod1.


#(3)
library(MASS)
lda.fit=lda(type~.,data=train)
lda.fit  #Prior probabilities of groups先验概率; Coefficients of linear discriminants线性方程系数
lda.pred=predict(lda.fit,test)
lda.pred$class  #预测的所属类的结果;后验概率为lda.pred$posterior

tab=table(lda.pred$class,test$type)  #预测值和真实值
tab    #混淆矩阵
erro=1-sum(diag(prop.table(tab)))    #计算误判率
erro

ldap1=prediction((as.numeric(lda.pred$class)-1),test$type) #预测值和真实值 
ldaperf1=performance(ldap1,"tpr","fpr")
plot(ldaperf1,colorize=TRUE,lwd=4)
#AUC:0.941
performance(ldap1, "auc")@y.values[[1]]

modelldaroc1=roc(test$type,(as.numeric(lda.pred$class)-1)) #真实值和预测值
plot(modelldaroc1, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

#(4)
lda.fit2=lda(type~Cl.thickness+Cell.shape+Marg.adhesion+Bare.nuclei+Bl.cromatin,data=train)
lda.fit2  #Prior probabilities of groups先验概率; Coefficients of linear discriminants线性方程系数
lda.pred2=predict(lda.fit2,test)
lda.pred2$class  #预测的所属类的结果;后验概率为lda.pred$posterior

tab2=table(lda.pred2$class,test$type)  #预测值和真实值
tab2    #混淆矩阵
erro2=1-sum(diag(prop.table(tab2)))    #计算误判率
erro2

ldap2=prediction((as.numeric(lda.pred2$class)-1),test$type) #预测值和真实值 
ldaperf2=performance(ldap2,"tpr","fpr")
plot(ldaperf2,colorize=TRUE,lwd=4)
#AUC:0.936
performance(ldap2, "auc")@y.values[[1]]

modelldaroc2=roc(test$type,(as.numeric(lda.pred2$class)-1)) #真实值和预测值
plot(modelldaroc2, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

#(5)
qda.fit=qda(type~.,data=train)
qda.fit  #Prior probabilities of groups先验概率
qda.pred=predict(qda.fit,test)
qda.pred$class  #预测的所属类的结果;后验概率为qda.pred$posterior

tabq1=table(qda.pred$class,test$type)  #预测值和真实值
tabq1    #混淆矩阵
erroq1=1-sum(diag(prop.table(tab)))    #计算误判率
erroq1

qdap1=prediction((as.numeric(qda.pred$class)-1),test$type) #预测值和真实值 
qdaperf1=performance(qdap1,"tpr","fpr")
plot(qdaperf1,colorize=TRUE,lwd=4)
#AUC:0.951
performance(qdap1, "auc")@y.values[[1]]

modelqdaroc1=roc(test$type,(as.numeric(qda.pred$class)-1)) #真实值和预测值
plot(modelqdaroc1, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

#(6)
#   MODEL               AUC
#glm full model         0.943
#glm smaller model      0.953
#LDA full model         0.941
#LDA smaller model      0.936
#QDA full model         0.951


###3
#Actually I've done the homework at the exact day it's assigned.
#So I wrote this myKNN function 2 days later when the notification announced. The result is about the same as the kknn library function.
#But I still reserve my codes written before because maybe they will be useful in the future.
#I can write a loop for k and h for myknn function, but that's not so efficient as the knn.cv or train.kknn in libraries to choose the best k and h.
#So I still choose them for searching best k and h, and use my function to do the result check, and make sure the myknn function I wrote is correct.
#These are the codes for both my function and the library functions.


myKNN_Gaussian=function(trainx,trainy,testx,testy,k=9,h=2,delta=0.5){
  
  M=dim(trainx)[1]
  N=dim(testx)[1]
  Xt=rbind(trainx,testx)
  
  Dtest=as.matrix(dist(Xt, method = "euclidean"))
  
  Dtest=Dtest[-c(1:M),]
  Pre=c()
  Error=c()

  for (l in 1:N) {
    Tsort=sort(Dtest[l,1:M], index.return = TRUE)
    index=Tsort$ix[1:k]
    Weight=exp((-0.5*((Tsort$x[1:k]/h)^2)))  ### you can change the kernel function here.
    Vote=trainy[Tsort$ix[1:k]]
    tent=data.frame("Weight"=Weight,"Vote"=Vote)
    tent2=aggregate(tent[, c("Weight")], list(Vote = tent$Vote), sum)
    
    if (dim(tent2)[1]==1){
      if (tent2[1,1]==1){
        if (delta==0){
          PreY=0
        }else{
          PreY=1
        }
      }else{
        PreY=0
      }
      
    }else{
      if (tent2$x[tent2$Vote==0]/(tent2$x[tent2$Vote==0]+tent2$x[tent2$Vote==1])>=delta){
        PreY=0
      }else{
        PreY=1
      }
    }
    
    ErrorY=PreY!=testy[l]
    Pre[l]=PreY
    Error[l]=ErrorY
  }
  return(Pre)
  
}


#(1)&(2)
library(class)
library(kknn)

trainx=train[,1:9]#训练集
testx=test[,1:9]#测试集
trainy=train[,10]
testy=test[,10]

### Check the best k and h at the same time:
############### kknn part in class library ###############

#测试不同方法,distance=2
# model.tkknn=train.kknn(type~.,train,kernel = c("rectangular","triangular","epanechnikov","biweight","triweight","cos","inv","gaussian","optimal"),distance=2,scale=T,kmax=30)
# model.tkknn$MISCLASS #显示错误率;$MEAN.ABS平均绝对误差;$MEAN.SQU均方误差
# model.tkknn #输出最优参数情况
# plot(model.tkknn)

#核函数为epanechnikov时，取distance=1-4进行测试
model.tkknn1=train.kknn(type~.,train,kernel = "gaussian",distance=1,scale=T,kmax=30)
model.tkknn1 #输出最优参数情况
plot(model.tkknn1)

model.tkknn2=train.kknn(type~.,train,kernel = "gaussian",distance=2,scale=T,kmax=30)
model.tkknn2 #输出最优参数情况
plot(model.tkknn2)

model.tkknn3=train.kknn(type~.,train,kernel = "gaussian",distance=3,scale=T,kmax=30)
model.tkknn3 #输出最优参数情况
plot(model.tkknn3)

model.tkknn4=train.kknn(type~.,train,kernel = "gaussian",distance=4,scale=T,kmax=30)
model.tkknn4 #输出最优参数情况
plot(model.tkknn4)

#There's no significant changes so we can take distance=2 and k=9

# knntab2=table(trainy,model.tkknn2$fitted.values[[9]])
# knntab2                            #混淆矩阵
# 1-sum(diag(prop.table(knntab2)))   #计算误判率

#采用最优参数做预测
model.kknn=kknn(type~.,train,testx,k=model.tkknn2$best.parameters$k,scale=T,distance=2,kernel=model.tkknn2$best.parameters$kernel)
# model.kknn$C #邻居的观测值号
# train[model.kknn$C,]#查看邻居
# model.kknn$D #邻居与它的距离
# model.kknn$W #邻居的权重
# model.kknn$CL #邻居的类别
# model.kknn$prob#预测的概率

#summary(model.kknn)
fit=fitted(model.kknn)
fit

knntab2=table(testy,fit)
knntab2                            #混淆矩阵
1-sum(diag(prop.table(knntab2)))   #计算误判率

############### myknn part to check ##########################
starttime=Sys.time()

myfit=myKNN_Gaussian(trainx,trainy,testx,testy,k=9,h=2)

knntabmy=table(testy,myfit)
knntabmy                            #混淆矩阵
1-sum(diag(prop.table(knntabmy)))   #计算误判率

endtime=Sys.time()
timeinterval=endtime-starttime
timeinterval

#The function I wrote is correct with the kknn function, but a bit slower.

#Then do the ROC manually and compare it with the picture in the library:

###################### ROC manually ##########################

deltalist=c(0.001,0.01,0.02,0.1,0.2,0.5,0.8,0.9,0.98,0.99,0.999)
TPR_sensitivity=c(1)
FPR_1_Sensitivity=c(1)
for (i in 1:length(deltalist)) {
  pred_tent=rep("1",291)
  pred_tent[which(model.kknn$prob[,1]>deltalist[i])]="0"
  pred_tent=as.numeric(pred_tent)
  knntab_tent=table(pred_tent,testy)  #预测值和真实值 
  TPR_sensitivity[i+1]=knntab_tent[1,1]/(knntab_tent[1,1]+knntab_tent[2,1])
  FPR_1_Sensitivity[i+1]=knntab_tent[1,2]/(knntab_tent[1,2]+knntab_tent[2,2])
}
TPR_sensitivity=append(TPR_sensitivity,0)
FPR_1_Sensitivity=append(FPR_1_Sensitivity,0)
plot(FPR_1_Sensitivity,TPR_sensitivity,type="b")


###################### compare with the library function ##########################

knnp2=prediction((as.numeric(fit)-1),testy) #预测值和真实值 
knnperf2=performance(knnp2,"tpr","fpr")
plot(knnperf2,colorize=TRUE,lwd=4)
#AUC:0.960
performance(knnp2, "auc")@y.values[[1]]

modelknnroc2=roc(testy,(as.numeric(fit)-1)) #真实值和预测值
plot(modelknnroc2, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)


############### knn part in class library ###############

#knn(trainx,testx,cl=trainy,k=2,prob=T)#example

#采用留一交叉验证寻找最优的k
error=rep(0,30)
for (i in 1:30){
  set.seed(1)#设置随机数种子为1
  cv1=knn.cv(trainx,cl=trainy, k=i, prob = TRUE)
  error[i]=sum(as.numeric(as.numeric(cv1)!=as.numeric(trainy)))/nrow(trainx)
  #错判率
}
error
plot(error,type="b",xlab="k")
#the best tuned K-value is K=6

#取k=6进行预测

predict1=knn(trainx,testx,cl=trainy,k=6,prob=T)
knntab1=table(testy,predict1)
knntab1                            #混淆矩阵
1-sum(diag(prop.table(knntab1)))   #计算误判率

knnp1=prediction((as.numeric(predict1)-1),testy) #预测值和真实值 
knnperf1=performance(knnp1,"tpr","fpr")
plot(knnperf1,colorize=TRUE,lwd=4)
#AUC:0.965
performance(knnp1, "auc")@y.values[[1]]

modelknnroc1=roc(testy,(as.numeric(predict1)-1)) #真实值和预测值
plot(modelknnroc1, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)


#(3)
#   MODEL               AUC
#glm full model         0.943
#glm smaller model      0.953
#LDA full model         0.941
#LDA smaller model      0.936
#QDA full model         0.951
#KNN rectangular k=6    0.965 ***√
#KNN Gaussian h=2 k=9   0.960 **



