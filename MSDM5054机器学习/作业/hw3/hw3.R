#(1)
#1
# The cubic spline model for a set of data points (x1, y1), (x2, y2), ..., (xn, yn) and knots ξ1, ξ2, ..., ξK can be written
# as follows:
# For each interval [ξi, ξi+1], we fit a cubic polynomial of the form:
# f_i(x) = a_i0 + a_i1*x + a_i2*x^2 + a_i3*x^3
#
# We then apply the continuity and linearity constraints at each knot ξi:
#
# Continuity Constraints:
# f_i(ξi) = f_i+1(ξi)
# f_i'(ξi) = f_i+1'(ξi)
# f_i''(ξi) = f_i+1''(ξi)
#
# And also Linearity Constraints at endpoints (-∞, ξ1) and (ξK, ∞).
# The objective function can be a least squares error.
# Now the problem becomes a linear regression problem with equality constraints.
# We can solve the coefficients a_i, a_i1, a_i2, a_i3, subject to these equality constraints.

#2
# See in the picture 详见手写证明过程 

#3
# See in the picture 详见手写证明过程

#4
# Piecewise polynomial regression divides the data set into multiple segments, 
# and uses a polynomial function to fit each segment. The fitting of each segment is performed independently; 
#
# Local polynomial regression is a non-parametric method that uses the neighbors near the point to fit the point,
# based on weights in the neighborhood, which is smoother but more complex.
#
# My understanding is: piecewise polynomial regression is suitable for situations
# where there are clear intervals between variables with different polynomial behavior;
# while local polynomial regression is more suitable for capturing smooth local changes in data that are not clearly segmented.

#5
# a. Lack of Global Information: Since local reference method regression only focuses on local data points,
# it usually cannot provide global information about the entire data set.
# 
# b. Non-Parametric Nature: Local polynomial regression is a non-parametric regression method that
# does not rely on a specific functional form. 
# Compared with traditional parametric models like linear regression, the parameters are less interpretable.
# 
# c. Bandwidth Selection: Bandwidth Selection further complicates interpretation.

#6
# For a local demonstration regression model, assume that we use the least squares method for demonstration:
# y_hat(x) = Σ(1~n) w_i(x)*y_i
# Bias(x) = E[y_hat(x)] - y
# Variance(x) = E[(y_hat(x) - E[y_hat(x)])^2]

# When increasing the bandwidth h: 
#
# Change in deviation:
# Increasing the bandwidth h makes the model smoother, it uses more neighbor data points to fit.
# This means that the predicted value y_hat(x) is more biased, reducing the model's ability to capture local details. 
# 
# Change in variance:
# Increasing the bandwidth h reduces the model's sensitivity to neighbor data points, thus reducing the model's variance. 
# When the bandwidth increases, the range of data points covered by the model's weight function w_i(x) increases, 
# and the model is relatively insensitive to changes in neighbor data points.
#
# Conclusion: h↑, Bias↑, Var↓

#7
# I think Regression tree is most similar to option (b) Piecewise constant regression:
# Regression tree and piecewise constant regression both achieve modeling of nonlinear relationships 
# by dividing the feature space into multiple regions and using different constant values for fitting.
#
# differences:
# Piecewise constant regression is a simple regression method that divides the feature space into several segments
# and fits a constant value in each segment. Its model is relatively simple, can only capture piecewise linear relationships, 
# and cannot handle the complex structure of the feature space.
# 
# Regression tree is a non-parametric supervised learning method that divides the feature space 
# into a series of rectangular regions and fits a constant value in each region.
# Regression trees are more flexible and can capture nonlinear relationships and interactions 
# by continuously dividing the feature space, so they perform well when dealing with nonlinear data and complex relationships. 
# And it can automatically determine the location of the division based on the distribution of the data, 
# while piecewise constant regression requires manually specifying the location of the division.

#8
# Increasing complexity reduces bias: the model is better able to capture approximate features and relationships 
# in the training data and can more accurately adapt to the complexity of the data, thus reducing bias.
#
# Increasing complexity increases variance: Model complexity causes the model to be very sensitive to small changes 
# in training data, and the difference in predictions on different training data sets increases, thus increasing variance.

#9
# Linear regression:
# If the true relationship between the predictor and response variables is indeed linear, 
# linear regression can accurately capture this relationship, with less bias, and provide a good fit to the true relationship.
# 
# Regression tree:
# Regression trees are more suitable for handling non-linear and piecewise relationships. 
# If the true relationship is linear, the regression tree may have difficulty capturing it accurately, 
# so the deviation can be large. In this case, the regression tree is often not as accurate as linear regression. 
# Even though regression trees can capture complex non-linear relationships, there can be high bias since the 
# decision boundaries created by a single tree structure may not exactly match the true underlying relationships.
# 
# Natural splines:
# Natural splines is a form of nonlinear regression that captures nonlinear patterns in data. 
# The deviation of natural splines depends on the complexity of the spline function and the number of nodes used. 
# When the number of nodes is limited or the complexity of the spline function is low, especially when the true relationship 
# is close to linear, natural splines may have higher deviation.
#
# To sum up, in general, regression tree has relatively the smallest bias if data nonlinear, or linear regression is better.

#10
# The variable importance measurement mainly uses "average decrease in accuracy" and "average decrease of Gini index".
#
# "The average reduction in accuracy" means that for each tree model: predict the cases outside the bag and 
# obtain the accuracy; then randomly arrange the values of a certain variable on the cases outside the bag, 
# and then bring them in the model to obtains the accuracy, and obtains the difference between the two accuracy rates.
# Average the accuracy differences of all trees on this variable to get the "average decrease in accuracy". 
#
# For each tree, IM_Gini(Rm) = ∑k!=k'(p^mk*p^mk') = K∑k=1(p^mk(1 ??? p^mk)) = 1 ??? K∑k=1(p^mk^2) 
#
# The average decrease in the Gini index refers to, for a tree, obtaining the decrease in the Gini index 
# caused by a certain variable acting as a splitting variable on each node;
# summing up the decreases in the Gini index of the variable in all trees and dividing Based on the number of trees,
# then the "average reduction of Gini index" is obtained.
#
# In the variable section/model selection part, we prefer variables with higher importance. 


#(2)
#1
df1=read.csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw3//trees.csv")
fit.1=lm(Volume ~ poly(Girth, 1, raw = TRUE), data = df1)
fit.2=lm(Volume ~ poly(Girth, 2, raw = TRUE), data = df1)
fit.3=lm(Volume ~ poly(Girth, 3, raw = TRUE), data = df1)
fit.4=lm(Volume ~ poly(Girth, 4, raw = TRUE), data = df1)

adj_rsq1=summary(fit.1)$adj.r.squared
adj_rsq2=summary(fit.2)$adj.r.squared
adj_rsq3=summary(fit.3)$adj.r.squared
adj_rsq4=summary(fit.4)$adj.r.squared

max(c(adj_rsq1, adj_rsq2, adj_rsq3, adj_rsq4))
which.max(c(adj_rsq1, adj_rsq2, adj_rsq3, adj_rsq4))
# deg = 2

## prediction on all range of age, and confidence bands
Girthlims=range(df1$Girth)
Girth.grid=seq(from=Girthlims[1],to=Girthlims[2])
preds=predict(fit.2,newdata=list(Girth=Girth.grid),se=TRUE)
se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)

par(mfrow=c(1,1))
plot(df1$Girth,df1$Volume,xlim=Girthlims ,cex=1,col="black")
title("Degree -2 Polynomial ",outer=F)
lines(Girth.grid,preds$fit,lwd=2,col="blue")
matlines(Girth.grid,se.bands,lwd=1,col="blue",lty=3)

#### choosing the model using 5-CV error:
library(boot)

cv.error=rep(0,4)
for (i in 1:4){
  set.seed(1)
  glm.fit=glm(Volume ~ poly(Girth,i, raw = TRUE),data=df1)
  cv.error[i]=cv.glm(df1,glm.fit,K=5)$delta[1]
}
cv.error
plot(1:4,cv.error,type='b',xlab="Degree of Polynomial",ylab="CV Error",main="5-fold CV")
#We still choose the i=2 model.

#2
Girth=df1$Girth
Volume=df1$Volume

fit=glm(I(Volume>30)~poly(Girth,2),data=df1,family=binomial)
preds=predict(fit,newdata=list(Girth=Girth.grid),se=T)              ## predict on all the age values
pfit=exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))

preds=predict(fit,newdata=list(Girth=Girth.grid),type="response", se=T)

##### plot the confidence bands
plot(Girth,I(Volume>30),xlim=Girthlims ,type="n")
points(jitter(Girth), I(Volume>30),cex=.5,pch="|",col =" darkgrey ")
lines(Girth.grid,pfit,lwd=2, col="blue")
matlines(Girth.grid,se.bands,lwd=1,col="blue",lty=3)

#3
library(splines)
fit2=lm(Volume~bs(Girth,knots=c(10,14,18),df=2),data=df1)
pred2=predict(fit2,newdata=list(Girth=Girth.grid),se=T)
plot(Girth,Volume,col="gray",main="Regression Spline on Selected Knots (deg=2)")
lines(Girth.grid,pred2$fit,lwd=2,col="blue")
lines(Girth.grid,pred2$fit+2*pred2$se ,lty="dashed",col="red")
lines(Girth.grid,pred2$fit-2*pred2$se ,lty="dashed",col="red")

#4
library(caret)
########################   10-fold CV of Smoothing Spline
X=data.frame(Girth=df1$Girth)
y=df1$Volume
flds=createFolds(1:31, k = 10, list = TRUE, returnTrain = FALSE)    ### random split into 10 folds
CVErr=rep(0,19)
for(k in 2:20){
  err=0
  for(fold in flds){
    fit=smooth.spline(X[-fold,], y[-fold],df=k)
    test_y=predict(fit,X[fold,])
    err=err+sum((test_y$y-y[fold])^2)
  }
  CVErr[k-1]=err/10
}
CVErr
plot(1:19,CVErr,type='b',xlab="Degrees of Freedom",ylab="CV Error",main="10-fold CV of Smoothing Spline")

plot(Girth,Volume,xlim=Girthlims,cex=.5,col="darkgrey",main=" Smoothing Spline ")
fit=smooth.spline(Girth,Volume,df=16)
fit2=smooth.spline(Girth,Volume,cv=TRUE)
fit2$df # degree=3.87138
lines(fit,col="red",lwd=2)
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("16 DF","3.87 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)

#5
library(gam)
gam.m3=gam(Volume~s(Girth,4)+s(Height,5),data=df1) 
par(mfrow=c(1,2))
plot(gam.m3, se=TRUE,col="blue") 

par(mfrow=c(1,1))


#(3)
#1
traindf=read.csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw3//audit_train.csv")
testdf=read.csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw3//audit_test.csv")

traindf=na.omit(traindf)
testdf=na.omit(testdf)

traindf$Risk = as.factor(traindf$Risk)
testdf$Risk = as.factor(testdf$Risk)

library(tree)
audittree=tree(Risk~.,traindf,control =tree.control(dim(traindf)[1],mindev=0.005,minsize=40))
summary(audittree)
plot(audittree)
text(audittree,pretty=0)

# Misclassification error rate: 0.06957 = 40 / 575 

predp=predict(audittree,testdf)

pred=rep(0,dim(testdf)[1])
pred[predp[,2]>0.5]=1

table(pred,testdf$Risk)
length(which(pred==testdf$Risk))/dim(testdf)[1] # accuracy = 0.9441624

#2
cv.audittree = cv.tree(audittree,FUN=prune.misclass)
names(cv.audittree)
cv.audittree

plot(cv.audittree$size ,cv.audittree$dev ,type="b",xlab="Size",ylab="CV Error")

# size = 5

audittree_pruned=prune.tree(audittree,best=5)
summary(audittree_pruned)
plot(audittree_pruned)
text(audittree_pruned,pretty=0)

predp2=predict(audittree_pruned,testdf)

pred2=rep(0,dim(testdf)[1])
pred2[predp2[,2]>0.5]=1

table(pred2,testdf$Risk)
length(which(pred2==testdf$Risk))/dim(testdf)[1] # accuracy = 0.9441624

#3
library(randomForest)

set.seed(1)
rf.audit=randomForest(Risk~.,data=traindf,mtry=13,ntree=25,importance=T,proximity=T,na.action=na.omit)   
rf.audit

# OOB estimate of  error rate: 9.91%
#     0   1 class.error
# 0 326  26  0.07386364
# 1  31 192  0.13901345



#4
error = c()

mlist = c(8,12,14,16,18)

for (m in mlist) {
  set.seed(1)
  rfmodel=randomForest(Risk~.,data=traindf,mtry=m,ntree=25,importance=T,proximity=T,na.action=na.omit)
  error=c(error,rfmodel$err.rate[25,1])
}

plot(mlist,error,type="b",xlab="m",ylab="Train Error") 

error # m = 8 min

set.seed(1)
rfmodel2 = randomForest(Risk~.,data=traindf,mtry=8,ntree=25,importance=T,proximity=T,na.action=na.omit)

yhat.rf = predict(rfmodel2,newdata=testdf,type="class")
table(yhat.rf,testdf$Risk)
length(which(yhat.rf==testdf$Risk))/dim(testdf)[1] # accuracy = 0.9695431

rfmodel2$importance # Score, Risk_D, TOTAL, Money_Value

#5
# single tree: accuracy = 0.9441624
# random forest: accuracy = 0.9695431 , which is better than a single tree. The model built on decision trees is less likely
# to overfit after selecting appropriate parameters, which can reduce errors and deviations, and improve accuracy.
# Due to the combination of multiple decision trees, diversity is effectively considered, and the combined result is
# better than the result of a single tree。

# summary(audittree_pruned):    "Score" "Money_Value" "TOTAL"  "District_Loss"
# rfmodel2$importance:           Score,  Money_Value , TOTAL ,  Risk_D
# 3 of them are the same.

#(4)
#a
































