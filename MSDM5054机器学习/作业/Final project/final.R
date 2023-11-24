
#(1)

#1
# Random Forest:
#
# The idea of random forest: On the basis of bagging (changing training samples), the diversity of the base learner is further enhanced
# by changing the modeling variables. Specifically: for each node of the base decision tree, from the change of the node randomly 
# select a subset containing k variables from the quantity set, and then select an optimal variable from this subset for branching.
#
# For each tree i = 1, · · · ,T: (1) Use the Bootstrap method to extract n sample observations from all training sample observations to 
# form the Bootstrap data set D*; (2) Based on data set D* Construct a tree hi and repeat the following steps for each node in the tree
# until the stopping rule is met; (3) Output a combination of T trees.
#
# The computational complexity of random forest is: T(O(nk log2(n)) + O(s) + O(n))
# The computational complexity of the base decision tree is O(nk log2(n)); The complexity of Bootstrap sampling and voting/averaging 
# is O(s); variables are randomly selected at the root node and intermediate nodes, with about n nodes, Therefore the complexity is O(n);
# There are T base decision trees in total.

# Gradient Boosting Trees:
# 
# Decision Tree (GBDT) is an additive model form: fm(x) = fm-1(x) + hm(x).
# Consider the squared loss function, The hm(x) generated at step m should be in the direction of the local maximum decrease of L 
# with respect to fm-1(x). In summary, at the mth step: hm(x) should be in the local direction described by the gradient
# -gm = y - fm-1(x) up. hm(x) should be a decision tree with εm = y - fm-1(x) as the dependent variable.
#
# The computational complexity of the decision tree can be expressed as O(TNMlog(M)), where T is the number of iterations.

#2
# Decision trees have strong interpretability, while random forests are relatively weak in model interpretability.

# Decision tree is a machine learning algorithm based on tree structure. Each node of the decision tree represents a feature 
# attribute, the branches of the node represent the value of the feature attribute, and the leaf nodes represent the final 
# classification or regression results. Due to the clear structure, we can directly observe the judgment conditions and branch 
# paths of each node to understand how the model makes predictions.

# Random forest is an ensemble learning method that consists of multiple decision trees. The final prediction result of 
# a random forest is obtained by voting or averaged by all decision trees. Each decision tree may adopt different features 
# and parameter settings, so the interpretability of the entire model becomes more difficult. In addition, random forest 
# introduces randomness in the construction process, including random selection of features and random sampling of data, 
# which also increases the difficulty of fully interpreting the model. Since the number of decision trees in a random forest 
# is large and the contribution of each decision tree is relatively small, it is difficult to map the prediction results of 
# the entire random forest to a single feature or decision.

#3
# Construct a Lagrangian function: Lagrange = L(Y, f) + λ(f1 + f2 + ... + fK), minimize Lagrange.
# We take the partial derivatives of f1, f2, ..., fK and λ and set them equal to zero to get the following system of equations:
# 
# exp((-Y·f*)/K) · (-Y1/K) + λ = 0
# exp((-Y·f*)/K) · (-Y2/K) + λ = 0
# ...
# exp((-Y·f*)/K) · (-YK/K) + λ = 0
# f*1 + f*2 + ... + f*K = 0
# 
# We can use numerical optimization methods to approximate the solution, such as gradient descent.
# The class probability can be expressed as P(Y = 1 | G = Gk) = P(G = Gk). Yk = 1 or Yk = -1/(K-1). therefore:
# P(Y = 1 | G = Gk) = P(G = Gk) = (1 + 1/(K-1)) * P(Yk = 1)
# P(Y = -1/(K-1) | G = Gk) = (1/(K-1)) * P(Yk = -1/(K-1))
# By definition we have ∑P(Yk = 1) = 1, therefore ∑P(G = Gk) = 1.
#
# When we minimize the loss function L(Y, f), the value of the loss function increases for misclassified samples and decreases 
# for correctly classified samples. This causes the weight of incorrectly classified samples to increase and the weight of 
# correctly classified samples to decrease in the next iteration.
# Similar to Adaboost, we can calculate the weighting factor for each sample based on the classification error. Specifically, 
# for sample i, we define the weight factor as: wi = exp((-Yi·f)/K). This weight factor is consistent with the form of 
# the loss function L(Y, f)

#4
# Suppose there is an optimal classification hyperplane whose distance to the left is greater than the distance to the right.
# In this case, consider two projection points on the optimal classification hyperplane, which are located on the boundaries of the left and right intervals respectively. 
# Let the projection point on the left be A and the projection point on the right be B.
# Since the distance on the left side is the largest distance on the right side, we can move the projection point A on the optimal 
# classification hyperplane along the direction of the normal arrangement, and at the same time move the projection point B to the 
# right until they are both located on their respective boundaries, instead of changing Classification results of data points.
# Doing this will cause the method of optimizing the classification hyperplane to change, but since we only made small adjustments, 
# this new hyperplane will still be able to correctly classify the data points of both categories.
# However, this contradicts the definition of a maximum margin classifier.

#5
# Contains three types of support vectors:
# 1.Points lying on hyperplanes L+1 and L-1.  (0 < λi < C and ξi = 0 );
# 
# 2.Points that fall within the interval and are correctly classified.   (λi = C and 0 < ξi ≤ 1);
# 
# 3.Points that are not correctly classified.   (λi = C and ξi > 1).

#6
# From the perspective of loss function plus penalty: ξi can be expressed as: ξi = max(0,1-yi(β^T * xi + β0)). This is Hinge Loss.
# Using hinge loss, the above objective can be rewritten as: min(β,β0){1∑n(1-yi(xi^T *β + β0)) + λ/2|β|^2))}

#7
# The fundamental difference between the two algorithms is that K-means is essentially unsupervised learning, 
# while KNN is supervised learning; K-means is a clustering algorithm, and KNN is a classification (or regression) algorithm.
# 
# KNN belongs to supervised learning, and the categories are known. By training and learning the data of known categories, 
# we can find the characteristics of these different categories, and then classify the unclassified data.
# 
# Kmeans belongs to unsupervised learning. It is not known in advance how many categories the data will be divided into, 
# and the data is aggregated into several groups through cluster analysis. Clustering does not require training and learning from the data.

#8
# Principal components analysis is an unsupervised technique that projects raw data into several high vertical directions 
# These high vertical directions are orthogonal, so the correlation of the projected data is very low or almost close to 0. 
# These feature transformations are linear.

# An autoencoder is an unsupervised artificial neural network that compresses data into lower dimensions and then reconstructs 
# the input. Autoencoders find lower-dimensional representations of data by removing noise and redundancy on important features.

# PCA can only perform linear transformations, while autoencoders can perform both linear and nonlinear transformations;

# The PCA algorithm is fast to calculate, while the autoencoder needs to be trained through the gradient descent algorithm, 
# so it takes longer time;

# PCA projects the data into several orthogonal directions, while the data dimensions are not necessarily orthogonal 
# after autoencoder dimensionality reduction;

# The only hyperparameter of PCA is the number of orthogonal vectors, while the hyperparameters of the autoencoder are 
# the structural parameters of the neural network;

# Autoencoders can also be used on complex, large data sets.

#9

#10




#(2)
#1
documents=read.table("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\20newsgroup\\documents.txt", header = FALSE)
groupnames=read.table("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\20newsgroup\\groupnames.txt", header = FALSE)
newsgroups=read.table("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\20newsgroup\\newsgroups.txt", header = FALSE)
wordlist=read.table("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\20newsgroup\\wordlist.txt", header = FALSE)

library(tidyr)
library(gtools)

tent=pivot_wider(documents, names_from = V2, values_from = V3, values_fill = 0)
df=as.data.frame(tent)
df=df[, -1]
sorted_cols=mixedsort(colnames(df))
df=df[,sorted_cols]

colnames(df)=wordlist$V1

newsgroups[newsgroups == 1]=groupnames[1,]
newsgroups[newsgroups == 2]=groupnames[2,]
newsgroups[newsgroups == 3]=groupnames[3,]
newsgroups[newsgroups == 4]=groupnames[4,]

df=as.data.frame(lapply(df, as.factor))

df$grouptype=newsgroups$V1

df$grouptype=as.factor(df$grouptype)

library(randomForest)
library(caret)

train_control=trainControl(method = "cv",number = 5)
param_grid=expand.grid(mtry = c(8, 10, 12))

set.seed(123)
rf_model1=train(x = df[,-101],y = df[,101], method = "rf", ntree = 150, trControl = train_control, tuneGrid = param_grid)
rf_model1$results
set.seed(123)
rf_model2=train(x = df[,-101],y = df[,101], method = "rf", ntree = 100, trControl = train_control, tuneGrid = param_grid)
rf_model2$results
set.seed(123)
rf_model3=train(x = df[,-101],y = df[,101], method = "rf", ntree = 200, trControl = train_control, tuneGrid = param_grid)
rf_model3$results

# We choose the ntree = 200 and mtry = 12 to get the lowest cv-error = 1 - 0.8148625 = 0.1851375

set.seed(123)
rf_model=randomForest(grouptype~., data = df, mtry=12, ntree=200, importance=T, proximity=T)
rf_model

# OOB estimate of  error rate: 18.65%
# Confusion matrix:
#          comp.* rec.* sci.* talk.* class.error
# comp.*   4142    73   195    195   0.1005429
# rec.*     300  2706   156    357   0.2310315
# sci.*     642   131  1488    396   0.4399699
# talk.*    258   126   200   4877   0.1069401

sorted_MeanDecreaseAccuracy=rf_model$importance[order(rf_model$importance[,5], decreasing = TRUE), ]
sorted_MeanDecreaseGini=rf_model$importance[order(rf_model$importance[,6], decreasing = TRUE), ]

sorted_MeanDecreaseAccuracy[1:10,]  # the same ↓
sorted_MeanDecreaseGini[1:10,]      # the same ↑

# So the ten most important keywords based on variable importance are:
# windows, god, christian, car, government, team, jews, graphics, space, religion.




#2
train_control2=trainControl(method = "cv", number = 5)
param_grid2=expand.grid(n.trees = c(100, 150, 200),interaction.depth = c(1,2,3),shrinkage = c(0.01,0.05,0.1),n.minobsinnode = c(15))

set.seed(123)
gbm_model=train(x = df[, -101],y = df[, 101],method = "gbm",trControl = train_control2,tuneGrid = param_grid2,verbose = FALSE)

gbm_model$results
gbm_model

# The final values used for the model were n.trees = 200, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 15.

library(gbm)
set.seed(123)
gbm_model2=gbm(grouptype~., data = df, distribution = "multinomial",n.trees=200, interaction.depth=3, shrinkage = 0.1)
summary(gbm_model2)

# So the ten most important keywords based on variable importance are:
# windows, god, christian, car, government, team, jews, graphics, space, gun (not religion).

predicted_classes=predict(gbm_model2, newdata = df, type = "response")
predicted_classes=colnames(predicted_classes)[apply(predicted_classes, 1, which.max)]

confusion_matrix=table(predicted_classes, df$grouptype)
confusion_matrix

# 计算错误率
error_rate=1 - sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Error rate:", error_rate))

# predicted_classes  comp.* rec.* sci.* talk.*
#           comp.*   4142   275   553    231
#           rec.*      62  2739   117    111
#           sci.*     217   160  1622    257
#           talk.*    184   345   365   4862
#
#            Error rate: 0.177133357960842


#3
#  Time : the gbm(Error rate: 0.177) is much slower and a bit more accurate than random forest(Error rate: 0.1851375).
#  Variable importance: the first 9 keywords are the same.

#4
library(MASS)

ctrl=trainControl(method = "cv", number = 5,verboseIter = FALSE)
lda_model=train(grouptype ~ ., data = df, method = "lda", trControl = ctrl)
lda_model
lda_model$results

# Accuracy: 0.7974388
# Misclassification Error = 1 - 0.7974388 = 0.2025612

#5
# I must reduce the dimensionality first otherwise qda will report an error: Error in qda.default(x, grouping, ...) : rank deficiency in group comp.*

df1=as.data.frame(tent)
df1=df1[, -1]
sorted_cols1=mixedsort(colnames(df1))
df1=df1[,sorted_cols1]
colnames(df1)=wordlist$V1

# df1=as.data.frame(lapply(df1, as.factor))

df1$grouptype=newsgroups$V1
df1$grouptype=as.factor(df1$grouptype)

# 进行主成分分析（PCA）
pca_result=prcomp(df1[, -which(names(df1) == "grouptype")], scale. = TRUE)  # 选择去除响应变量后的预测变量列

# 选择保留的主成分数量或方差百分比
# 这里以保留方差百分比为例，比如保留累积方差达到90%的主成分
variance_threshold=0.9
cumulative_variance=cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
num_components=which(cumulative_variance >= variance_threshold)[1]

# 使用选定的主成分数量进行降维
reduced_data=as.data.frame(predict(pca_result, newdata = df1)[, 1:num_components])

# 将响应变量添加回降维后的数据框
reduced_data$grouptype=df$grouptype

# 训练 QDA 模型
qda_model=train(grouptype ~ ., data = reduced_data, method = "qda", trControl = ctrl)
qda_model
qda_model$results

# Accuracy: 0.7710264
# Misclassification Error = 1 - 0.7710264 = 0.2289736


#6
library(e1071)

set.seed(1)
tune.out=tune(svm, grouptype~.,data=df,kernel="linear",scale=TRUE,ranges=list(cost=c(1,5,10,15,20,25,30)),cross = 5)
tune.out$performances
tune.out$best.performance

# Accuracy: 0.8090754
# Misclassification Error = 0.1909246


# set.seed(1)
# tune.out=tune(svm, grouptype~.,data=df, kernel="radial",ranges=list(cost=c(0.1,1,10,100),gamma=c(0.5,1,2,3,4)),cross = 5)
# summary(tune.out)


#7
#
#      MODEL        Accuracy   Time cost to train models
#
#  Random Forest   0.8148625     Middle
#       GBM        0.8228666     Large
#       LDA        0.7974388     Small
#       QDA        0.7710264     Small
#       SVM        0.8090754     Large

# The GBM has the best accuracy, however it needs the most time to train the model. Random Forest is better.
# The SVM also takes lots of time while its performance is not so good as Random Forest.

write.csv(df, file = "C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\group_data.csv", row.names = FALSE)



#(4)

























