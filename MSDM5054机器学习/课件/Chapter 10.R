################### 
###                              R codes for Chapter 10: Unsupervised Learning
###################


library(ISLR)

#############################################################################################################################
######   Principal Component Analysis
#############################################################################################################################

## In this lab, we perform PCA on the USArrests data set, which is part of the base R package. The rows of the data 
## set contain the 50 states, in alphabetical order.
states=row.names(USArrests)
states

## The columns of the data set contain the four variables.
names(USArrests )

## We first briefly examine the data. We notice that the variables have vastly different means.
apply(USArrests , 2, mean)

## Note that the apply() function allows us to apply a function—in this case, the mean() function—to each row or column of 
## the data set. The second input here denotes whether we wish to compute the mean of the rows, 1, or the columns, 2. We 
## see that there are on average three times as many rapes as murders, and more than eight times as many assaults as rapes.
## We can also examine the variances of the four variables using the apply() function.
apply(USArrests , 2, var)

## Not surprisingly, the variables also have vastly different variances: the UrbanPop variable measures the percentage of 
## the population in each state living in an urban area, which is not a comparable number to the num- ber of rapes in each 
## state per 100,000 individuals. If we failed to scale the variables before performing PCA, then most of the principal 
## components that we observed would be driven by the Assault variable, since it has by far the largest mean and variance. 
## Thus, it is important to standardize the variables to have mean zero and standard deviation one before performing PCA.

## We now perform principal components analysis using the prcomp() func- tion, which is one of several functions in R that perform PCA.
pr.out=prcomp(USArrests, scale=TRUE)


## By default, the prcomp() function centers the variables to have mean zero. By using the option scale=TRUE, we scale the 
## variables to have standard deviation one. The output from prcomp() contains a number of useful quan- tities.
names(pr.out)


## The center and scale components correspond to the means and standard deviations of the variables that were used for s
## caling prior to implementing PCA.
pr.out$center
pr.out$scale



## The rotation matrix provides the principal component loadings; each col- umn of pr.out$rotation contains the 
## corresponding principal component loading vector
pr.out$rotation


## We see that there are four distinct principal components. This is to be expected because there are in general 
## min(n − 1, p) informative principal components in a data set with n observations and p variables.


## Using the prcomp() function, we do not need to explicitly multiply the data by the principal component loading vectors 
## in order to obtain the principal component score vectors. Rather the 50 ×4 matrix x has as its columns the principal 
## component score vectors. That is, the kth column is the kth principal component score vector.
dim(pr.out$x)


## We can plot the first two principal components as follows:
biplot(pr.out, scale=0)
##The scale=0 argument to biplot() ensures that the arrows are scaled to represent the loadings; other values for scale 
## give slightly different biplots with different interpretations.



## Recall that the principal components are only unique up to a sign change, so we can reproduce Figure 10.1 by making a few small changes:
pr.out$rotation=-pr.out$rotation  
pr.out$x=-pr.out$x
biplot(pr.out, scale=0)


## The prcomp() function also outputs the standard deviation of each prin- cipal component. For instance, on the USArrests 
## data set, we can access these standard deviations as follows:
pr.out$sdev


## The variance explained by each principal component is obtained by squar- ing these
pr.var=pr.out$sdev^2
pr.var



## To compute the proportion of variance explained by each principal compo- nent, we simply divide the variance explained 
## by each principal component by the total variance explained by all four principal components:
pve=pr.var/sum(pr.var)
pve


## We see that the first principal component explains 62.0 % of the variance in the data, the next principal component 
## explains 24.7 % of the variance, and so forth. We can plot the PVE explained by each component, as well as the cumulative PVE, as follows:
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained ", ylim=c(0,1),type="b")
plot(cumsum(pve), xlab="Principal Component ", ylab=" Cumulative Proportion of Variance Explained ", ylim=c(0,1), type="b")



## Note that the function cumsum() computes the cumulative sum of the elements of a numeric vector. For instance:
a=c(1,2,8,-3) 
cumsum (a)




#############################################################################################################################
######   K-means Clustering
#############################################################################################################################

## The function kmeans() performs K-means clustering in R. We begin with a simple simulated example in which there truly 
## are two clusters in the data: the first 25 observations have a mean shift relative to the next 25 observations.
set.seed (2)
x=matrix(rnorm(50*2), ncol=2)
x[1:25,1]=x[1:25,1]+3
x[1:25,2]=x[1:25,2]-4

## We now perform K-means clustering with K = 2.
km.out=kmeans(x,2,nstart=20)

## The cluster assignments of the 50 observations are contained in km.out$cluster.
km.out$cluster


## The K-means clustering perfectly separated the observations into two clus- ters even though we did not supply any group 
## information to kmeans(). We can plot the data, with each observation colored according to its cluster assignment.
plot(x, col=(km.out$cluster +1), main="K-Means Clustering Results with K=2", xlab="", ylab="", pch=20, cex=2)
## Here the observations can be easily plotted because they are two-dimensional. If there were more than two variables 
## then we could instead perform PCA and plot the first two principal components score vectors.


## In this example, we knew that there really were two clusters because we generated the data. However, for real data, 
## in general we do not know the true number of clusters. We could instead have performed K-means clustering on this 
## example with K = 3.
set.seed (4)
km.out=kmeans(x,3,nstart=20)
km.out
## When K = 3, K-means clustering splits up the two clusters.


## To run the kmeans() function in R with multiple initial cluster assign- ments, we use the nstart argument. If a value 
## of nstart greater than one is used, then K-means clustering will be performed using multiple random assignments in 
## Step 1 of Algorithm 10.1, and the kmeans() function will report only the best results. Here we compare using nstart=1 
## to nstart=20.
set.seed (3)
km.out=kmeans(x,3,nstart=1)
km.out$tot.withinss
km.out=kmeans(x,3,nstart=20)
km.out$tot.withinss
## Note that km.out$tot.withinss is the total within-cluster sum of squares, which we seek to minimize by performing 
## K-means clustering (Equation 10.11). The individual within-cluster sum-of-squares are contained in the vector 
## km.out$withinss.


## We strongly recommend always running K-means clustering with a large value of nstart, such as 20 or 50, since otherwise 
## an undesirable local optimum may be obtained.
  

## When performing K-means clustering, in addition to using multiple ini- tial cluster assignments, it is also important 
## to set a random seed using the set.seed() function. This way, the initial cluster assignments in Step 1 can be 
## replicated, and the K-means output will be fully reproducible.





#############################################################################################################################
######  Hierarchical Clustering
#############################################################################################################################

## The hclust() function implements hierarchical clustering in R. In the fol- lowing example we use the data from 
## Section 10.5.1 to plot the hierarchical clustering dendrogram using complete, single, and average linkage cluster- ing, 
## with Euclidean distance as the dissimilarity measure. We begin by clustering observations using complete linkage. 
## The dist() function is used to compute the 50 ×50 inter-observation Euclidean distance matrix.
hc.complete=hclust(dist(x), method="complete")

## We could just as easily perform hierarchical clustering with average or single linkage instead:
hc.average=hclust(dist(x), method="average")  
hc.single=hclust(dist(x), method="single")


## We can now plot the dendrograms obtained using the usual plot() function. The numbers at the bottom of the plot identify each observation.
par(mfrow=c(1,3))
plot(hc.complete,main="Complete Linkage", xlab="", sub="",cex =.9)
plot(hc.average , main="Average Linkage", xlab="", sub="",cex =.9)
plot(hc.single , main="Single Linkage", xlab="", sub="",cex =.9)


## To determine the cluster labels for each observation associated with a given cut of the dendrogram, we can use the 
## cutree() function:
cutree(hc.complete, 2)
cutree(hc.complete, 2)
cutree(hc.single,2)


## For this data, complete and average linkage generally separate the observa- tions into their correct groups. 
## However, single linkage identifies one point as belonging to its own cluster. A more sensible answer is obtained when 
## four clusters are selected, although there are still two singletons.
cutree(hc.single , 4)


## To scale the variables before performing hierarchical clustering of the observations, we use the scale() function:
xsc=scale(x)
plot(hclust(dist(xsc), method="complete"), main="Hierarchical Clustering with Scaled Features ")


## Correlation-based distance can be computed using the as.dist() func- tion, which converts an arbitrary square symmetric 
## matrix into a form that the hclust() function recognizes as a distance matrix. However, this only makes sense for data 
## with at least three features since the absolute corre- lation between any two observations with measurements on two 
## features is always 1. Hence, we will cluster a three-dimensional data set.
x=matrix(rnorm(30*3), ncol=3)
dd=as.dist(1-cor(t(x)))
plot(hclust(dd, method="complete"), main="Complete Linkage with Correlation -Based Distance", xlab="", sub="")





#############################################################################################################################
######  NCI60 Data Example
#############################################################################################################################

## Unsupervised techniques are often used in the analysis of genomic data. In particular, PCA and hierarchical clustering 
## are popular tools. We illus- trate these techniques on the NCI60 cancer cell line microarray data, which consists of 
## 6,830 gene expression measurements on 64 cancer cell lines.
library(ISLR)
nci.labs=NCI60$labs 
nci.data=NCI60$data


## Each cell line is labeled with a cancer type. We do not make use of the cancer types in performing PCA and clustering, 
## as these are unsupervised techniques. But after performing PCA and clustering, we will check to see the extent to which 
## these cancer types agree with the results of these unsupervised techniques.


## We begin by examining the cancer types for the cell lines.
nci.labs[1:4]
table(nci.labs)


################################
###    PCA on NCI60 Data
#################################

## We first perform PCA on the data after scaling the variables (genes) to have standard deviation one, although one could reasonably argue that it is better not to scale the genes.
pr.out=prcomp(nci.data, scale=TRUE)


## We now plot the first few principal component score vectors, in order to visualize the data. The observations (cell 
## lines) corresponding to a given cancer type will be plotted in the same color, so that we can see to what extent the 
## observations within a cancer type are similar to each other. We first create a simple function that assigns a distinct 
## color to each element of a numeric vector. The function will be used to assign a color to each of the 64 cell lines, 
## based on the cancer type to which it corresponds.
Cols=function(vec){
  cols=rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])}
## Note that the rainbow() function takes as its argument a positive integer, and returns a vector containing that number of distinct colors. 

## We now can plot the principal component score vectors.
par(mfrow=c(1,2))
plot(pr.out$x[,1:2], col=Cols(nci.labs), pch=19,xlab="Z1",ylab="Z2")
plot(pr.out$x[,c(1,3)], col=Cols(nci.labs), pch=19,xlab="Z1",ylab="Z3")


## The resulting plots are shown in Figure 10.15. On the whole, cell lines corresponding to a single cancer type do tend 
## to have similar values on the first few principal component score vectors. This indicates that cell lines from the 
## same cancer type tend to have pretty similar gene expression levels.


## We can obtain a summary of the proportion of variance explained (PVE) of the first few principal components using the 
## summary() method for a prcomp object (we have truncated the printout):
summary(pr.out)


## Using the plot() function, we can also plot the variance explained by the first few principal components.
plot(pr.out)


## Note that the height of each bar in the bar plot is given by squaring the corresponding element of pr.out$sdev. 
## However, it is more informative to rainbow() plot the PVE of each principal component (i.e. a scree plot) and the 
## cu- mulative PVE of each principal component. This can be done with just a little work.
pve=100*pr.out$sdev^2/sum(pr.out$sdev^2)
par(mfrow=c(1,2))
plot(pve, type="o", ylab="PVE", xlab="Principal Component",col ="blue")
plot(cumsum(pve), type="o", ylab="Cumulative PVE", xlab="Principal Component ", col ="brown3")




################################
###    Clustering the Observations of the NCI60 Data
#################################

## We now proceed to hierarchically cluster the cell lines in the NCI60 data, with the goal of finding out whether or 
## not the observations cluster into distinct types of cancer. To begin, we standardize the variables to have mean zero 
## and standard deviation one. As mentioned earlier, this step is optional and should be performed only if we want each 
## gene to be on the same scale.
sd.data=scale(nci.data)

## We now perform hierarchical clustering of the observations using complete, single, and average linkage. Euclidean 
## distance is used as the dissimilarity measure.
par(mfrow=c(1,3))
data.dist=dist(sd.data)
plot(hclust(data.dist), labels=nci.labs, main="Complete Linkage", xlab="", sub="",ylab="")
plot(hclust(data.dist, method="average"), labels=nci.labs, main="Average Linkage", xlab="", sub="",ylab="")
plot(hclust(data.dist, method="single"), labels=nci.labs,main="Single Linkage", xlab="", sub="",ylab="")
## e see that the choice of linkage certainly does affect the results obtained. Typically, single linkage will tend to 
## yield trailing clusters: very large clusters onto which individual observa- tions attach one-by-one. On the other hand, 
## complete and average linkage tend to yield more balanced, attractive clusters. For this reason, complete and average 
## linkage are generally preferred to single linkage. Clearly cell lines within a single cancer type do tend to cluster 
## together, although the clustering is not perfect. We will use complete linkage hierarchical cluster- ing for the analysis that follows.

## We can cut the dendrogram at the height that will yield a particular number of clusters, say four:
hc.out=hclust(dist(sd.data)) 
hc.clusters=cutree(hc.out,4) 
table(hc.clusters,nci.labs)


## There are some clear patterns. All the leukemia cell lines fall in cluster 3, while the breast cancer cell lines are 
## spread out over three different clusters. We can plot the cut on the dendrogram that produces these four clusters:
par(mfrow=c(1,1))
plot(hc.out, labels=nci.labs) 
abline(h=139, col="red")
## The abline() function draws a straight line on top of any existing plot in R. The argument h=139 plots a horizontal 
## line at height 139 on the den- drogram; this is the height that results in four distinct clusters. It is easy to verify 
## that the resulting clusters are the same as the ones we obtained using cutree(hc.out,4).


## We claimed earlier in Section 10.3.2 that K-means clustering and hier- archical clustering with the dendrogram cut to 
## obtain the same number of clusters can yield very different results. How do these NCI60 hierarchical clustering results 
## compare to what we get if we perform K-means clustering with K = 4?
set.seed (2)
km.out=kmeans(sd.data, 4, nstart=20)
km.clusters=km.out$cluster
table(km.clusters ,hc.clusters )
## We see that the four clusters obtained using hierarchical clustering and K- means clustering are somewhat different. 
## Cluster 2 in K-means clustering is identical to cluster 3 in hierarchical clustering. However, the other clusters
## differ: for instance, cluster 4 in K-means clustering contains a portion of the observations assigned to cluster 1 by 
## hierarchical clustering, as well as all of the observations assigned to cluster 2 by hierarchical clustering.


## Rather than performing hierarchical clustering on the entire data matrix, we can simply perform hierarchical clustering 
## on the first few principal component score vectors, as follows:
hc.out=hclust(dist(pr.out$x[,1:5]))
plot(hc.out, labels=nci.labs, main="Hier. Clust. on First Five Score Vectors ")
table(cutree(hc.out,4), nci.labs)
## Not surprisingly, these results are different from the ones that we obtained when we performed hierarchical clustering 
## on the full data set. Sometimes performing clustering on the first few principal component score vectors can give better
## results than performing clustering on the full data. In this situation, we might view the principal component step as 
## one of denois- ing the data. We could also perform K-means clustering on the first few principal component score vectors
## rather than the full data set.









