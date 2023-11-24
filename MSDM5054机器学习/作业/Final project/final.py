import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

#(3)
#1
def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)

def k_means(df, k, iteration):
    #初始化簇心向量
    index = random.sample(list(range(len(df))), k)
    vectors = []
    for i in index:
        vectors.append(list(df.loc[i,].values))
    
    #初始化类别
    labels = []
    for i in range(len(df)):
        labels.append(-1)
    
    while(iteration > 0):
        #初始化簇
        C = []
        for i in range(k):
            C.append([])
        for labelIndex, item in enumerate(df.to_numpy()):
            classIndex = -1
            minDist = 1e6
            for i, point in enumerate(vectors):
                dist = getEuclidean(item, point)
                if(dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        
        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = df.shape[1]
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        
        iteration -= 1
    return C, labels

#2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import time

df0 = pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\group_data.csv")
df = df0.iloc[:,0:100]

scaler = StandardScaler()

# 对数据进行标准化
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

start_time = time.time()

# 创建 PCA 模型并进行主成分分析
pca = PCA(n_components=4)
principal_components = pca.fit_transform(scaled_df)

# 将主成分数据转换为数据框
pc_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2','PC3','PC4'])

random.seed(123)
C, labels = k_means(pc_df, 4, 20)

pc_df.loc[:,'grouptype']=labels

df0['grouptype'] = df0['grouptype'].replace('comp.*', 2)
df0['grouptype'] = df0['grouptype'].replace('talk.*', 3)
df0['grouptype'] = df0['grouptype'].replace('sci.*', 0)
df0['grouptype'] = df0['grouptype'].replace('rec.*', 1)

# 计算混淆矩阵
cm = confusion_matrix(df0.loc[:,'grouptype'], labels)
print(cm)

# 计算误判率
misclassification_rate = (np.sum(cm) - np.trace(cm)) / np.sum(cm)

print("misclassification_rate: "+ str(misclassification_rate))

end_time = time.time()
execution_time = end_time - start_time
print(f"Total cost time：{execution_time:.4f} seconds")


#3
df0 = pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\group_data.csv")
df = df0.iloc[:,0:100]

scaler = StandardScaler()

# 对数据进行标准化
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

start_time = time.time()

# 创建 PCA 模型并进行主成分分析
pca = PCA(n_components=5)
principal_components = pca.fit_transform(scaled_df)

# 将主成分数据转换为数据框
pc_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2','PC3','PC4','PC5'])

random.seed(123)
C, labels = k_means(pc_df, 4, 20)

pc_df.loc[:,'grouptype']=labels

df0['grouptype'] = df0['grouptype'].replace('comp.*', 2)
df0['grouptype'] = df0['grouptype'].replace('talk.*', 3)
df0['grouptype'] = df0['grouptype'].replace('sci.*', 0)
df0['grouptype'] = df0['grouptype'].replace('rec.*', 1)

# 计算混淆矩阵
cm = confusion_matrix(df0.loc[:,'grouptype'], labels)
print(cm)

# 计算误判率
misclassification_rate = (np.sum(cm) - np.trace(cm)) / np.sum(cm)

print("misclassification_rate: "+ str(misclassification_rate))

end_time = time.time()
execution_time = end_time - start_time
print(f"Total cost time：{execution_time:.4f} seconds")


#4
#      MODEL        Accuracy   Time cost to train models
#
#  Random Forest   0.8148625     Middle
#       GBM        0.8228666     Large
#       LDA        0.7974388     Small
#       QDA        0.7710264     Small
#       SVM        0.8090754     Large
#     K-means      0.4881788     Small


#5
# 使用 PCA 将数据投影到前三个主成分上
pca = PCA(n_components=3)
df_pca = pca.fit_transform(scaled_df)

# 绘制投影图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], marker='o')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Projection of Data onto First Three Principal Components')

ax.view_init(elev=10, azim=-15)

plt.show()

# We can see the 4 clusters' structure.






