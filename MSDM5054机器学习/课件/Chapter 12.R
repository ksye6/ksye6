############################################################
########
########    Self-Training with SSC library
########
############################################################

library(ssc)
data(wine) # load the Wine dataset
wine[1:10,]
summary(wine)

wine$Wine<-as.factor(wine$Wine)
pairs(wine, main="Scatterplot Matrix",pch=19)    ## ScatterPlot Matrix


cls <- which(colnames(wine) == "Wine")
x <- wine[, -cls] # instances without classes
y <- wine[, cls] # the classes
x <- scale(x) # scale the attributes for distance calculations set.seed(3)

# Use 50% of instances for training
tra.idx <- sample(x = length(y), size = ceiling(length(y) * 0.5)) 
xtrain <- x[tra.idx,] # training instances
ytrain <- y[tra.idx] # classes of training instances

# Use 70% of train instances as unlabeled set
tra.na.idx <- sample(x = length(tra.idx),
                     size = ceiling(length(tra.idx) * 0.7))
ytrain[tra.na.idx] <- NA # remove class of unlabeled instances

# Use the other 50% of instances for inductive test
tst.idx <- setdiff(1:length(y), tra.idx)
xitest <- x[tst.idx,] # test instances
yitest <- y[tst.idx] # classes of instances in xitest


# Use the unlabeled examples for transductive test
xttest <- x[tra.idx[tra.na.idx],] # transductive test instances 
yttest <- y[tra.idx[tra.na.idx]] # classes of instances in xttest

# computing distance and kernel matrices
dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE)) 
ditest <- as.matrix(proxy::dist(x = xitest, y = xtrain, method = "euclidean",by_rows = TRUE))
ktrain <- as.matrix(exp(- 0.048 * dtrain^2)) 
kitest <- as.matrix(exp(- 0.048 * ditest^2))



#######  Training
library(caret)
m.selft1 <- selfTraining(x = xtrain, y = ytrain, learner = knn3, learner.pars = list(k = 1), pred = "predict")

m.selft2 <- selfTraining(x = dtrain, y = ytrain, x.inst = FALSE, learner = oneNN, pred = "predict", pred.pars = list(type = "prob"))

library(kernlab)
m.selft3 <- selfTraining(x = ktrain, y = ytrain, x.inst = FALSE, learner = ksvm,
                         learner.pars = list(kernel = "matrix", prob.model = TRUE), pred = function(m, k)
                           predict(m, as.kernelMatrix(k[, SVindex(m)]), type = "probabilities")
)


####### Inductive Test
p.selft1 <- predict(m.selft1, xitest)
table(p.selft1,yitest)
length(which(p.selft1==yitest))/length(yitest)  ##accuracy

p.self2 <- predict(m.selft2, ditest[,m.selft2$instances.index])
table(p.self2,yitest)
length(which(p.self2==yitest))/length(yitest)  ##accuracy


p.selft3<-predict(m.selft3,as.kernelMatrix(kitest[,m.selft3$instances.index]))
table(p.selft3,yitest)
length(which(p.selft3==yitest))/length(yitest)  ##accuracy










####################### Using only supervised learning
xl.train<-xtrain[-tra.na.idx,]    ## using only labeled data for training
yl.train<-ytrain[-tra.na.idx]

l.train<-cbind(xl.train,yl.train)

### Supervised 1-KNN
knn.fit<-class::knn(xl.train,xitest,yl.train,k=1)  ###k=1
table(knn.fit, yitest)   ### confusion matrix on test data
length(which(knn.fit==yitest))/length(yitest)  ##accuracy



### Supervised 3-KNN
knn.fit<-class::knn(xl.train,xitest,yl.train,k=3)  ###k=3
table(knn.fit, yitest)   ### confusion matrix on test data
length(which(knn.fit==yitest))/length(yitest)  ##accuracy


### LDA
library(MASS)
itest<-data.frame(cbind(xitest,yitest))
l.train<-data.frame(l.train)
lda.fit<-lda(yl.train~.,data=l.train)
lda.fit.pred<-predict(lda.fit,itest,type="response")           ## multi-class,use defualt threshold
table(lda.fit.pred$class,itest$yitest)
length(which(lda.fit.pred$class==itest$yitest))/length(yitest)  ##accuracy



########### Supervised SVM
library(e1071)
l.train$yl.train<-as.factor(l.train$yl.train)
svmfit=svm(yl.train~., data=l.train, kernel="linear", cost=10,scale=FALSE)          ## Linear Kernel
ypred=predict(svmfit ,itest)
table(ypred,itest$yitest)
length(which(ypred==itest$yitest))/length(yitest)



########### Supervised SVM
library(e1071)
l.train$yl.train<-as.factor(l.train$yl.train)
svmfit=svm(yl.train~., data=l.train, cost=5,scale=TRUE,kernel="radial", gamma=0.1)          ## Gaussian Kernel
ypred=predict(svmfit ,itest)
table(ypred,itest$yitest)
length(which(ypred==itest$yitest))/length(yitest)



