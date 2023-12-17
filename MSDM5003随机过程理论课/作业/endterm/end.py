# -*- coding: gbk -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import imageio
from scipy.optimize import curve_fit

class Board(object):
    
    def __init__(self, L):
        self.L = int(L) # �߳� >=100
        self.K = 0.1 # Ӱ����ʵ�Ȩ�� <2
        self.b = 2 # ��ƭ���� 1~2
        self.strategy = np.random.randint(2, size=(self.L,self.L)) # ��ʼ���������
        self.valuematrix = None # �����˵ı����������
        self.nextstrategy = None # �����˵���һ�β��Ծ���
        self.strategydict = {} # ����Ԫ��(K,b),ֵ��һ���б����ȫ��ÿ��ģ��Ĳ���
        self.densitydict = {} # ����Ԫ��(K,b),ֵ��һ���б����ȫ��ÿ��ģ��ĺ������ܶ�
        self.simulationtimes = 18 # ���ģ�����
        
        self.filepath = "C:\\Users\\�����\\Desktop\\results" # gif����·��
        self.duration=200 # gif���,��λ�Ǻ���
        
        self.blist=np.arange(1, 2.1, 0.02) # b�����б�
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

# # ���²��Ծ��̶�KΪ�趨ֵ��LΪ�߳�100
# 
# # ����һ��b
# bb = Board(200)  # ����Ϊ200���Ѿ������ˣ�
# bb.total_simu()  # ���趨�õ�b����ģ��
# bb.densitydict   # ģ������ڵ�c�ܶȱ仯
# bb.create_gif(bb.K,bb.b)  # ���趨·���ļ����£����趨�õ�b����gifͼ
# 
# # ����ȫ��b
# aa = Board(200)  # ����Ϊ200���Ѿ������ˣ�
# aa.super_simu()  # ���趨�õ�blist�������b����ģ��
# aa.densitydict   # ����b��ģ������ڵ�c�ܶȱ仯�ֵ�
# aa.c_b_plot()    # ÿ��bѡ�����һ��ģ����ܶȽ����Ϊ�����꣬��c_bͼ
# aa.super_gif()   # ������b����gifͼ

################################################################

# ��Ϻ�����MΪģ��
def FIT(M):
    M.densitylist_5 = np.array(M.densitylist_5)
    
    index_fit1 = np.abs(M.densitylist_5 - 0.9).argmin()
    index_fit2 = np.abs(M.densitylist_5 - 0.1).argmin()
    
    range_f2 = [index_fit2 - 10,index_fit2]
    range_f1 = [index_fit1,index_fit1 + 10]

    b_fit2 = M.blist[range_f2[0]:range_f2[1]]
    b_c2 = b_fit2[-1]
    c_fit2 = M.densitylist[range_f2[0]:range_f2[1]]
    c_5_fit2 = M.densitylist_5[range_f2[0]:range_f2[1]]
    
    print(b_fit2)
    
    b_fit1 = M.blist[range_f1[0]:range_f1[1]]
    b_c1 = b_fit1[0]
    c_fit1 = M.densitylist[range_f1[0]:range_f1[1]]
    c_5_fit1 = M.densitylist_5[range_f1[0]:range_f1[1]]
    
    print(b_fit1)
    
    # ��Ϲ�ʽ
    def fit2_function(b, beta):
        return (b_c2 - b)**beta
    
    def fit1_function(b, beta):
        return 1 - (b - b_c1)**beta
    
    optimized_params_fit2, params_covariance_fit2 = curve_fit(fit2_function, b_fit2, c_5_fit2)
    print("b_c2 = ",b_c2," ;�� = ",optimized_params_fit2[0])
    
    optimized_params_fit1, params_covariance_fit1 = curve_fit(fit1_function, b_fit1, c_5_fit1)
    print("b_c1 = ",b_c1," ;�� = ",optimized_params_fit1[0])

    # �ۺ���ʾ
    
    fig = plt.figure(dpi=300)
    plt.plot(M.blist, M.densitylist_5, marker='o')
    plt.plot(b_fit1, 1 - ((b_fit1 - b_c1)**optimized_params_fit1[0]), color='green',label='fit1')
    plt.plot(b_fit2, ((b_c2 - b_fit2)**optimized_params_fit2[0]), color='red',label='fit2')
    plt.title("Density of cooperators about b ; K = " + str(M.K))
    plt.xlabel("b")
    plt.ylabel("c")
    plt.xlim(1, 2.1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

################################################################
#1 L=100,K=0.1

model1 = Board(100)  # ����Ϊ100
np.random.seed(123)
model1.super_simu()  # ���趨�õ�blist�������b����ģ��
model1.c_b_plot()    # ÿ��bѡ�����һ��ģ����ܶȽ����Ϊ�����꣬��c_bͼ
# model1.super_gif()   # ������b����gifͼ

FIT(model1)

################################################################
#2 L=100,K=0.5
model2 = Board(100)  # ����Ϊ100
model2.update_K(0.5)
np.random.seed(123)
model2.super_simu()  # ���趨�õ�blist�������b����ģ��
model2.c_b_plot()    # ÿ��bѡ�����һ��ģ����ܶȽ����Ϊ�����꣬��c_bͼ
# model2.super_gif()   # ������b����gifͼ

FIT(model2)

#####################################################################
class Board2(object):
    
    def __init__(self, L):
        self.L = int(L) # �߳� >=100
        self.K = 0.02 # Ӱ����ʵ�Ȩ�� <2
        self.b = 1.35 # ��ƭ���� 1~2
        self.strategy = np.random.randint(2, size=(self.L,self.L)) # ��ʼ���������
        self.valuematrix = None # �����˵ı����������
        self.nextstrategy = None # �����˵���һ�β��Ծ���
        self.strategydict = {} # ����Ԫ��(K,b),ֵ��һ���б����ȫ��ÿ��ģ��Ĳ���
        self.densitydict = {} # ����Ԫ��(K,b),ֵ��һ���б����ȫ��ÿ��ģ��ĺ������ܶ�
        self.simulationtimes = 20 # ���ģ�����
        
        self.filepath = "C:\\Users\\�����\\Desktop\\results2" # gif����·��
        self.duration=200 # gif���,��λ�Ǻ���
        
        self.blist=np.arange(1, 2.1, 0.008) # b�����б�
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
        topleft = None
        topright = None
        bottomleft = None
        bottomright = None
        
        if row > 0:
            top = strategy[row-1, col]
        
        if row < L - 1:
            bottom = strategy[row+1, col]
        
        if col > 0:
            left = strategy[row, col-1]
        
        if col < L - 1:
            right = strategy[row, col+1]
        
        if row > 0 and col > 0:
            topleft = strategy[row-1, col-1]
        
        if row > 0 and col < L - 1:
            topright = strategy[row-1, col+1]
        
        if row < L - 1 and col > 0:
            bottomleft = strategy[row+1, col-1]
        
        if row < L - 1 and col < L - 1:
            bottomright = strategy[row+1, col+1]
        
        tent=np.array([top,bottom,left,right,topleft,topright,bottomleft,bottomright])
        
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
        topleft = None
        topright = None
        bottomleft = None
        bottomright = None
        
        if row > 0:
            top = valuematrix[row-1, col]
        
        if row < L - 1:
            bottom = valuematrix[row+1, col]
        
        if col > 0:
            left = valuematrix[row, col-1]
        
        if col < L - 1:
            right = valuematrix[row, col+1]
        
        if row > 0 and col > 0:
            topleft = valuematrix[row-1, col-1]
        
        if row > 0 and col < L - 1:
            topright = valuematrix[row-1, col+1]
        
        if row < L - 1 and col > 0:
            bottomleft = valuematrix[row+1, col-1]
        
        if row < L - 1 and col < L - 1:
            bottomright = valuematrix[row+1, col+1]
        
        tent=np.array([top,bottom,left,right,topleft,topright,bottomleft,bottomright])
        
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

#3 L=100,K=0.02
model3 = Board2(100)  # ����Ϊ100
np.random.seed(123)
model3.super_simu()  # ���趨�õ�blist�������b����ģ��
model3.c_b_plot()    # ÿ��bѡ�����һ��ģ����ܶȽ����Ϊ�����꣬��c_bͼ
# model3.super_gif()   # ������b����gifͼ

fig = plt.figure(dpi=300)
plt.plot(model3.blist, model3.densitylist, marker='o')
plt.title("Density of cooperators about b ; K = " + str(model3.K))
plt.xlabel("b")
plt.ylabel("c")
plt.xlim(1, 2.1)
plt.ylim(0, 1)
plt.show()

FIT(model3)

#####################################################################

#4 L=100,K=0.5
model4 = Board2(100)  # ����Ϊ100
model4.update_K(0.5)
np.random.seed(123)
model4.super_simu()  # ���趨�õ�blist�������b����ģ��
model4.c_b_plot()    # ÿ��bѡ�����һ��ģ����ܶȽ����Ϊ�����꣬��c_bͼ
# model4.super_gif()   # ������b����gifͼ

fig = plt.figure(dpi=300)
plt.plot(model4.blist, model4.densitylist, marker='o')
plt.title("Density of cooperators about b ; K = " + str(model4.K))
plt.xlabel("b")
plt.ylabel("c")
plt.xlim(1, 2.1)
plt.ylim(0, 1)
plt.show()

FIT(model4)





