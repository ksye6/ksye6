#(4)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw3//Hitters_train.csv")
test = pd.read_csv("C://Users//张铭韬//Desktop//学业//港科大//MSDM5054机器学习//作业//hw3//Hitters_test.csv")

train = train[["Years", "Hits", "RBI","Walks", "PutOuts", "Runs","Salary"]]
test = test[["Years", "Hits", "RBI","Walks", "PutOuts", "Runs","Salary"]]

train.dropna(inplace=True)
test.dropna(inplace=True)

import random
import collections
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from collections import deque
 
class TreeNode:
    def __init__(self,labels_idx=None,left=None,right=None,split_idx=None,is_discrete=None,split_value=None,father=None) -> None:

        self.labels_idx = labels_idx     # 训练集的label对应的下标
        self.left = left                 # 左子树
        self.right = right               # 右子树
        self.split_idx = split_idx       # 划分特征对应的下标
        self.is_discrete = is_discrete   # 是否离散
        self.split_value = split_value   # 划分点
        self.father = father             # 父节点
 
class RegressionTree:
    
    def __init__(self,data,labels,is_discrete,validate_ratio=0.1):

        self.data = np.array(data)
        self.labels=np.array(labels)
        self.feature_num = self.data.shape[1]
        self.is_discrete = is_discrete
        self.validate_ratio = validate_ratio
        self.leaves = []
        if validate_ratio>0:
            all_index = range(data.shape[0])
            self.train_idx,self.test_idx = train_test_split(all_index,test_size=validate_ratio)
            self.validate_data = self.data[self.test_idx,:]
            self.validate_label = self.labels[self.test_idx]
            self.train_data = self.data[self.train_idx,:]
            self.train_label = self.labels[self.train_idx]
    
    def sum_std(self,x):
        
        return np.sum(np.abs(x-np.mean(x)))/len(x)  

    def choose_feature(self,x,left_labels):

        std_list = []
        split_value_list = []
        for i in range(x.shape[1]):
            final_split_value,final_sum_std=self.calc_std(x[:,i],self.is_discrete[i],left_labels)
            std_list.append(final_sum_std)
            split_value_list.append(final_split_value)
        idx = np.argmin(std_list)
        return idx,split_value_list[idx]
    
    def calc_std(self,feature,is_discrete,labels):

        final_sum_std = float("inf")
        final_split_value = 0
        idx = range(len(feature))
        feature_with_idx = np.c_[idx,feature]
        labels = np.array(labels)
        if is_discrete:
            values = list(set(feature))
            idx_dict = {v:[] for v in values}
            for i,fea in feature_with_idx:
                idx_dict[fea].append(i)
            for v in values:
                anti_idx = [i for i in idx if i not in idx_dict[v]]
                left = labels[idx_dict[v]]
                right = labels[anti_idx]
                if left.shape[0]==0 or right.shape[0] == 0:
                    continue
                sum_std = self.sum_std(left)+self.sum_std(right)
                if sum_std<final_sum_std:
                    final_sum_std = sum_std
                    final_split_value = v
        else:
            feature_with_idx = feature_with_idx[feature_with_idx[:,1].argsort()]
            feature = feature_with_idx[:,1]
            idx = feature_with_idx[:,0]
            for i in range(len(feature)-1):
                if feature[i]==feature[i+1]:
                    continue
                split_value = (feature[i]+feature[i+1])/2
                idx_left = idx[:i+1]
                idx_right = idx[i+1:]
                sum_std = self.sum_std(labels[idx_left.astype('int64')])+self.sum_std(labels[idx_right.astype('int64')])
                if sum_std<final_sum_std:
                    final_sum_std = sum_std
                    final_split_value = split_value
                    
        return final_split_value,final_sum_std
    
    def generate_tree(self,idxs,min_ratio):

        root = TreeNode(labels_idx=idxs)
        
        if len(idxs)/self.data.shape[0]<=min_ratio:
            return root
        
        idx,split_value = self.choose_feature(self.data[idxs,:],self.labels[idxs])
        root.split_value = split_value
        root.split_idx = idx
        left_idxs = []
        right_idxs = []
        
        if self.is_discrete[idx]:
            for i in idxs:
                if self.data[i,idx] != split_value:
                    right_idxs.append(i)
                else:
                    left_idxs.append(i)
        
        else:
            for i in idxs:
                if self.data[i,idx] <= split_value:
                    right_idxs.append(i)
                else:
                    left_idxs.append(i)
        
        left_idxs = np.array(left_idxs)
        right_idxs = np.array(right_idxs)
        root.left = self.generate_tree(left_idxs,min_ratio)
        
        if root.left:
            root.left.father = root
        
        root.right = self.generate_tree(right_idxs,min_ratio)
        
        if root.right:
            root.right.father = root
        
        return root
 
    def train(self,max_depth = 0,min_ratio=0.05):

        if self.validate_ratio>0:
            idx = self.train_idx
        else:
            idx = range(len(self.labels))
        
        self.tree = self.generate_tree(idx,min_ratio)
        
        # 后剪枝
        if self.validate_ratio>0:
            self.find_leaves(self.tree)
            nodes = deque(self.leaves)
            while len(nodes)>0:
                n=len(nodes)
                for _ in range(n):
                    node = nodes.popleft()
                    if not node.father:
                        nodes = []
                        break
                    valid_pred = self.predict(self.validate_data)
                    mse_before = self.get_mse(valid_pred,self.validate_label)
                    backup_left = node.father.left
                    backup_right= node.father.right
                    node.father.left = None
                    node.father.right = None
                    valid_pred = self.predict(self.validate_data)
                    mse_after = self.get_mse(valid_pred,self.validate_label)
                    if mse_after>mse_before:
                        node.father.left = node.father.left
                        node.father.right = node.father.right
                    else:
                        nodes.append(node.father)
        # 树深
        if max_depth>0:
            nodes = deque([self.tree])
            d=1
            while len(nodes)>0 and d<max_depth:
                n = len(nodes)
                for _ in range(n):
                    node = nodes.popleft()
                    if node.left:
                        nodes.append(node.left)
                    if node.right:
                        nodes.append(node.right)
                d += 1
            if len(nodes)>0:
                for node in nodes:
                    node.left=None
                    node.right=None
        
    def find_leaves(self,node):

        if not node.left and not node.right:
            self.leaves.append(node)
            return None
        else:
            if node.left:
                self.find_leaves(node.left)
            if node.right:
                self.find_leaves(node.right)
        
    def predict_one(self,x,node=None):

        if node == None:
            node = self.tree
        while node.left and node.right:
            idx = node.split_idx
            if self.is_discrete[idx]:
                if x[idx]==node.split_value:
                    node = node.left
                else:
                    node = node.right
            else:
                if x[idx]>node.split_value:
                    node = node.right
                else:
                    node = node.left
 
        res_idx = node.labels_idx
        return np.mean(self.labels[res_idx])
    
    def predict(self,x,node=None):

        x = np.array(x)
        predicts = []
        for i in range(x.shape[0]):
            res = self.predict_one(x[i,:],node)
            predicts.append(res)
        return predicts
 
    def get_mse(self,y_pred,y_true):

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        return np.mean(np.square(y_pred-y_true))

x_train = train.iloc[:,0:6].values
x_test = test.iloc[:,0:6].values
y_train = train.iloc[:,6].values
y_test = test.iloc[:,6].values

rt = RegressionTree(x_train,y_train,is_discrete=[False,False,False,False,False,False],validate_ratio=0.1)

rt.train(max_depth=15,min_ratio=0.05) # 限制深度和最小v_gain (实际函数已经进行剪枝)
res = rt.predict(x_test)
rt.get_mse(res,y_test)

# ############ 对比库 #############
# from sklearn.tree import DecisionTreeRegressor
# 
# tes = DecisionTreeRegressor()
# tes.fit(x_train,y_train)
# res2 = tes.predict(x_test)
# rt.get_mse(res2,y_test)

