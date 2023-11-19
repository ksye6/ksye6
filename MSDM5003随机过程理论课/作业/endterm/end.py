import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


L=300  # �߳� >=200
b=1.5  # ��ƭ���� 1~2
K=0.8  # Ӱ����ʵ�Ȩ�� <2

strategy = np.random.randint(2, size=(L,L))

# ȫ���ھӵĲ��ԣ���������б�
def neighbour_strategy(row,col,strategy):
  
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
def count_value(mystrategy,nei):
  
    tent=0
    
    if mystrategy==0 and nei==1:
        tent+=b
    
    if mystrategy==1 and nei==1:
        tent+=1
    
    return tent

# �Լ���ȫ���ھӵ����棬���ȫ����������V
def strategy_value(row,col,strategy,neighbours):
    mystra = strategy[row,col]
    totalvalue = 0
    for neistra in neighbours:
        totalvalue+=count_value(mystra,neistra)
    totalvalue += mystra
    return totalvalue

# �����˵��������
def value_matrix(strategy):
    valuematrix = np.zeros((L, L),int)
    for i in range(L):
        for j in range(L):
            valuematrix[i,j]+=strategy_value(i,j,strategy,neighbour_strategy(i,j,strategy))
    
    return valuematrix

# ����ƴ���ж��ٸ���ѡ��Է��Ĳ���
def possibility_to_choose(EX,EY):
    p = 1/(1+np.exp(-(EY-EX)/K))
    return p

# ȫ���ھӵ����棬˳���������ͬ����������б�
def neighbour_value(row,col,valuematrix):
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
def final_choice(neighbourvalues,myvalue,neighbourstrategies,mystrategy):
    
    n = len(neighbourvalues)
    plist = [possibility_to_choose(myvalue,EY)/n for EY in neighbourvalues]
    plist.append(1-sum(plist))
    tentlist = list(neighbourstrategies)
    tentlist.append(mystrategy)
    
    return np.random.choice(tentlist, p=plist)

# ȫ���˵�һ��ģ���������ģ���Ĳ��Ծ���
def simulation(strategy):
    
    valuematrix=value_matrix(strategy)
    
    new = np.zeros((L, L),int)
    
    for i in range(L):
        for j in range(L):
            new[i,j]=final_choice(neighbour_value(0,0,valuematrix),valuematrix[0,0],neighbour_strategy(0,0,strategy),strategy[0,0])
    
    return new

# N��ģ�⣬�������ÿ��ģ���������ܶȵ��б�
def total_simu(strategy,times):
    nextchoices=strategy
    c0 = list()
    c0.append(np.sum(nextchoices)/L**2)
    for i in range(times):
        nextchoices = simulation(nextchoices)
        c0.append(np.sum(nextchoices)/L**2)
    
    return c0

changes = total_simu(strategy,5)
print(changes)


aa=np.random.randint(2, size=(L,L))

print(aa)
print(np.sum(aa)/L**2)
aa=simulation(aa)


