# -*- coding: gbk -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import imageio

class Board(object):
    
    def __init__(self, L):
        self.L = int(L) # 边长 >=100
        self.K = 0.5 # 影响概率的权重 <2
        self.b = 2 # 欺骗收益 1~2
        self.strategy = np.random.randint(2, size=(self.L,self.L)) # 初始化随机策略
        self.valuematrix = None # 所有人的本次收益矩阵
        self.nextstrategy = None # 所有人的下一次策略矩阵
        self.strategydict = {} # 键是元组(K,b),值是一个列表包含全部每次模拟的策略
        self.densitydict = {} # 键是元组(K,b),值是一个列表包含全部每次模拟的合作者密度
        self.simulationtimes = 18 # 最多模拟次数
        
        self.filepath = "C:\\Users\\张铭韬\\Desktop\\results" # gif保存路径
        self.duration=200 # gif间隔,单位是毫秒
        
        self.blist=np.arange(1, 2.05, 0.02) # b收益列表
        self.densitylist=list() # 对每个b，选取最后一次模拟的合作者密度作为最终density结果，初始为空列表
        self.densitylist_5=None # 5点近似
    
    # 更新b
    def update_b(self, bb):
        self.b = bb
    
    # 更新K
    def update_K(self, KK):
        self.K = KK
    
    # 更新策略状态，前进一步
    def update_strategy(self):
        self.strategy = self.nextstrategy
    
    # 全部邻居的策略，输出策略列表
    def neighbour_strategy(self,row,col):
        
        L = self.L
        strategy = self.strategy
        top = None
        bottom = None
        left = None
        right = None
        
        if row > 0:
            top = strategy[row-1, col]
        
        if row < L - 1:
            bottom = strategy[row+1, col]
        
        if col > 0:
            left = strategy[row, col-1]
        
        if col < L - 1:
            right = strategy[row, col+1]
        
        tent=np.array([top,bottom,left,right])
        
        return tent[tent != None]
    
    # 自己的单次收益判定，输出单次收益数额V0
    def count_value(self,mystrategy,nei):
      
        tent=0
        
        if mystrategy==0 and nei==1:
            tent+=self.b
        
        if mystrategy==1 and nei==1:
            tent+=1
        
        return tent
    
    # 自己对全部邻居的收益，输出全部收益数额V
    def strategy_value(self,row,col,neighbours):
        
        strategy = self.strategy
        mystra = strategy[row,col]
        totalvalue = 0
        for neistra in neighbours:
            totalvalue+=self.count_value(mystra,neistra)
        totalvalue += mystra
        return totalvalue
    
    # 获得所有人的收益矩阵
    def value_matrix(self):
        L = self.L
        valuematrix = np.zeros((L, L),int)
        for i in range(L):
            for j in range(L):
                neighbours = self.neighbour_strategy(i,j)
                valuematrix[i,j]+=self.strategy_value(i,j,neighbours)
        
        return valuematrix
    
    # 单次拼点有多少概率选择对方的策略
    def possibility_to_choose(self,EX,EY):
        p = 1/(1+np.exp(-(EY-EX)/self.K))
        return p
    
    # 全部邻居的收益，顺序与策略相同，输出收益列表
    def neighbour_value(self,row,col):
        L = self.L
        valuematrix = self.valuematrix
        top = None
        bottom = None
        left = None
        right = None
        
        if row > 0:
            top = valuematrix[row-1, col]
        
        if row < L - 1:
            bottom = valuematrix[row+1, col]
        
        if col > 0:
            left = valuematrix[row, col-1]
        
        if col < L - 1:
            right = valuematrix[row, col+1]
        
        tent=np.array([top,bottom,left,right])
        
        return tent[tent != None]
    
    # 自己的最终策略选择，输出0（欺骗）或1（合作）
    def final_choice(self,neighbourvalues,myvalue,neighbourstrategies,mystrategy):
        
        n = len(neighbourvalues)
        plist = [self.possibility_to_choose(myvalue,EY)/n for EY in neighbourvalues]
        plist.append(1-sum(plist))
        tentlist = list(neighbourstrategies)
        tentlist.append(mystrategy)
        
        return np.random.choice(tentlist, p=plist)
    
    # 全部人的一次模拟结果
    def simulation(self,strategy):
        
        self.valuematrix = self.value_matrix()
        valuematrix=self.valuematrix
        L=self.L
        new = np.zeros((L, L),int)
        
        for i in range(L):
            for j in range(L):
                neighbourvalues = self.neighbour_value(i,j)
                neighbourstrategies = self.neighbour_strategy(i,j)
                new[i,j]=self.final_choice(neighbourvalues,valuematrix[i,j],neighbourstrategies,strategy[i,j])
        
        return new
    
    # 固定K和b，N次模拟，更新合作者密度,以及所有策略的列表
    def total_simu(self):
        L = self.L
        times = self.simulationtimes
        tentdensity = np.sum(self.strategy)/L**2
        
        self.densitydict[(self.K,self.b)] = list()
        self.strategydict[(self.K,self.b)] = list()
        
        self.densitydict[(self.K,self.b)].append(tentdensity)
        self.strategydict[(self.K,self.b)].append(self.strategy)
        
        for i in range(times):
            self.nextstrategy = self.simulation(self.strategy)
            self.update_strategy()
            self.densitydict[(self.K,self.b)].append(np.sum(self.strategy)/L**2)
            self.strategydict[(self.K,self.b)].append(self.strategy)
    
    # 固定K,对所有b进行N次模拟，更新最终合作者密度,以及所有策略的列表，以及最终合作者密度列表
    def super_simu(self):
        for bb in self.blist:
            self.update_b(bb)
            self.total_simu()
            self.densitylist.append(self.densitydict[(self.K, self.b)][-1]) #选取最后的合作者密度作为结果
            
            self.strategy = np.random.randint(2, size=(self.L,self.L)) # 初始化随机策略
            self.valuematrix = None
            self.nextstrategy = None
        
        self.densitylist_5 = self.change_to_five()
    
    # super_simu后，输出c_b密度图
    def c_b_plot(self):
        
        
        
        fig = plt.figure(dpi=300)
        
        plt.plot(self.blist, self.densitylist_5, marker='o')
        
        plt.title("Density of cooperators about b ; K = " + str(self.K))
        plt.xlabel("b")
        plt.ylabel("c")
        plt.xlim(1, 2.1)
        plt.ylim(0, 1)
        
        plt.show()  
    
    # 生成一张gif，名字为L_K_b.gif
    def create_gif(self,K,b):
        
        strategy_changes = self.strategydict[(K,b)]
        images = []
        L = self.L
        for frame in strategy_changes:
            image = Image.new("RGB", (L, L), "white")
            for i in range(L):
                for j in range(L):
                    pixel_value = frame[i,j]
                    color = "black" if pixel_value == 0 else "white"
                    image.paste(color, (j, i, (j + 1), (i + 1)))
            image = image.resize((800, 800))
            images.append(image)
        
        full_path = self.filepath + "\\" + str(L) + "_" + str(K) + "_" + str(round(b,2)) + ".gif"
        images[0].save(full_path, save_all=True, append_images=images[1:], optimize=False, duration=self.duration, loop=1)
    
    # 固定K，生成所有blist的gif
    def super_gif(self):
        K = self.K
        for bb in self.blist:
            self.create_gif(K,bb)
    
    # density变为5点近似
    def change_to_five(self):
        
        original_array = self.densitylist
        result_array = []
        
        # 遍历原始数组中的每个元素
        for i in range(len(original_array)):
            # 确定最近五个点的范围
            start_index = max(0, i - 2)
            end_index = min(len(original_array) - 1, i + 2)
            average = sum(original_array[start_index:end_index+1]) / (end_index - start_index + 1)
            # 将平均值添加到结果数组中
            result_array.append(average)
        
        return result_array
    

#############################  测试  ###########################

# 以下测试均固定K为设定值，L为边长100

# 测试一个b
bb = Board(100)  # 长度为100，已经很慢了（
bb.total_simu()  # 对设定好的b进行模拟
bb.densitydict   # 模拟次数内的c密度变化
bb.create_gif(bb.K,bb.b)  # 在设定路径文件夹下，对设定好的b生成gif图

# 测试全部b
aa = Board(100)  # 长度为100，已经很慢了（
aa.super_simu()  # 对设定好的blist里的所有b进行模拟
aa.densitydict   # 所有b的模拟次数内的c密度变化字典
aa.c_b_plot()    # 每个b选择最后一次模拟的密度结果作为纵坐标，画c_b图
aa.super_gif()   # 对所有b生成gif图


