# -*- coding: gbk -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import imageio

class Board(object):
    
    def __init__(self, L):
        self.L = int(L) # �߳� >=100
        self.K = 0.5 # Ӱ����ʵ�Ȩ�� <2
        self.b = 2 # ��ƭ���� 1~2
        self.strategy = np.random.randint(2, size=(self.L,self.L)) # ��ʼ���������
        self.valuematrix = None # �����˵ı����������
        self.nextstrategy = None # �����˵���һ�β��Ծ���
        self.strategydict = {} # ����Ԫ��(K,b),ֵ��һ���б����ȫ��ÿ��ģ��Ĳ���
        self.densitydict = {} # ����Ԫ��(K,b),ֵ��һ���б����ȫ��ÿ��ģ��ĺ������ܶ�
        self.simulationtimes = 18 # ���ģ�����
        
        self.filepath = "C:\\Users\\�����\\Desktop\\results" # gif����·��
        self.duration=200 # gif���,��λ�Ǻ���
        
        self.blist=np.arange(1, 2.05, 0.02) # b�����б�
        self.densitylist=list() # ��ÿ��b��ѡȡ���һ��ģ��ĺ������ܶ���Ϊ����density�������ʼΪ���б�
        self.densitylist_5=None # 5�����
    
    # ����b
    def update_b(self, bb):
        self.b = bb
    
    # ����K
    def update_K(self, KK):
        self.K = KK
    
    # ���²���״̬��ǰ��һ��
    def update_strategy(self):
        self.strategy = self.nextstrategy
    
    # ȫ���ھӵĲ��ԣ���������б�
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
    
    # �Լ��ĵ��������ж������������������V0
    def count_value(self,mystrategy,nei):
      
        tent=0
        
        if mystrategy==0 and nei==1:
            tent+=self.b
        
        if mystrategy==1 and nei==1:
            tent+=1
        
        return tent
    
    # �Լ���ȫ���ھӵ����棬���ȫ����������V
    def strategy_value(self,row,col,neighbours):
        
        strategy = self.strategy
        mystra = strategy[row,col]
        totalvalue = 0
        for neistra in neighbours:
            totalvalue+=self.count_value(mystra,neistra)
        totalvalue += mystra
        return totalvalue
    
    # ��������˵��������
    def value_matrix(self):
        L = self.L
        valuematrix = np.zeros((L, L),int)
        for i in range(L):
            for j in range(L):
                neighbours = self.neighbour_strategy(i,j)
                valuematrix[i,j]+=self.strategy_value(i,j,neighbours)
        
        return valuematrix
    
    # ����ƴ���ж��ٸ���ѡ��Է��Ĳ���
    def possibility_to_choose(self,EX,EY):
        p = 1/(1+np.exp(-(EY-EX)/self.K))
        return p
    
    # ȫ���ھӵ����棬˳���������ͬ����������б�
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
    
    # �Լ������ղ���ѡ�����0����ƭ����1��������
    def final_choice(self,neighbourvalues,myvalue,neighbourstrategies,mystrategy):
        
        n = len(neighbourvalues)
        plist = [self.possibility_to_choose(myvalue,EY)/n for EY in neighbourvalues]
        plist.append(1-sum(plist))
        tentlist = list(neighbourstrategies)
        tentlist.append(mystrategy)
        
        return np.random.choice(tentlist, p=plist)
    
    # ȫ���˵�һ��ģ����
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
    
    # �̶�K��b��N��ģ�⣬���º������ܶ�,�Լ����в��Ե��б�
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
    
    # �̶�K,������b����N��ģ�⣬�������պ������ܶ�,�Լ����в��Ե��б��Լ����պ������ܶ��б�
    def super_simu(self):
        for bb in self.blist:
            self.update_b(bb)
            self.total_simu()
            self.densitylist.append(self.densitydict[(self.K, self.b)][-1]) #ѡȡ���ĺ������ܶ���Ϊ���
            
            self.strategy = np.random.randint(2, size=(self.L,self.L)) # ��ʼ���������
            self.valuematrix = None
            self.nextstrategy = None
        
        self.densitylist_5 = self.change_to_five()
    
    # super_simu�����c_b�ܶ�ͼ
    def c_b_plot(self):
        
        
        
        fig = plt.figure(dpi=300)
        
        plt.plot(self.blist, self.densitylist_5, marker='o')
        
        plt.title("Density of cooperators about b ; K = " + str(self.K))
        plt.xlabel("b")
        plt.ylabel("c")
        plt.xlim(1, 2.1)
        plt.ylim(0, 1)
        
        plt.show()  
    
    # ����һ��gif������ΪL_K_b.gif
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
    
    # �̶�K����������blist��gif
    def super_gif(self):
        K = self.K
        for bb in self.blist:
            self.create_gif(K,bb)
    
    # density��Ϊ5�����
    def change_to_five(self):
        
        original_array = self.densitylist
        result_array = []
        
        # ����ԭʼ�����е�ÿ��Ԫ��
        for i in range(len(original_array)):
            # ȷ����������ķ�Χ
            start_index = max(0, i - 2)
            end_index = min(len(original_array) - 1, i + 2)
            average = sum(original_array[start_index:end_index+1]) / (end_index - start_index + 1)
            # ��ƽ��ֵ��ӵ����������
            result_array.append(average)
        
        return result_array
    

#############################  ����  ###########################

# ���²��Ծ��̶�KΪ�趨ֵ��LΪ�߳�100

# ����һ��b
bb = Board(100)  # ����Ϊ100���Ѿ������ˣ�
bb.total_simu()  # ���趨�õ�b����ģ��
bb.densitydict   # ģ������ڵ�c�ܶȱ仯
bb.create_gif(bb.K,bb.b)  # ���趨·���ļ����£����趨�õ�b����gifͼ

# ����ȫ��b
aa = Board(100)  # ����Ϊ100���Ѿ������ˣ�
aa.super_simu()  # ���趨�õ�blist�������b����ģ��
aa.densitydict   # ����b��ģ������ڵ�c�ܶȱ仯�ֵ�
aa.c_b_plot()    # ÿ��bѡ�����һ��ģ����ܶȽ����Ϊ�����꣬��c_bͼ
aa.super_gif()   # ������b����gifͼ


