import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

nsam = 10
Nlist = [num for num in range(21, 121, 10)]
x = 32/np.array(Nlist)

class Game(object):

    def __init__(self,m=5,s=2,N=21):
        self.m=m
        self.s=s
        self.N=N
        self.nrow=s**m
        self.iteration=1200
        self.iteq=200
        self.steps = 0
        self.t=list() # 迭代列表
        self.memorystate = format(0, '05b')  # 初始市场状态
        self.strategies = None # 所有策略名字
        self.chosen_strategy = None # 所有人每次选择的策略名字
        self.choices = None # 每个人的选择：0卖，1买
        self.agentvalue = None # 每个人的收益
        self.df = None # 详细策略内容
        self.winchoice = None # 少数方
        self.buyerlist = list() #1
        self.var = None
    
    def init_game(self):
        # 行索引
        self.row_index = [format(i, '05b') for i in range(self.nrow)]
        
        # 创建列索引
        self.col_index_10 = random.sample(range(2**self.nrow), self.N*self.s)
        self.col_index_2 = [format(i, '032b') for i in self.col_index_10]
        
        # 创建数据集
        data = [[int(col[i]) for col in self.col_index_2] for i in range(self.nrow)]
        
        # 创建DataFrame
        df = pd.DataFrame(data, index=self.row_index, columns=self.col_index_10)
        
        # 增加名为"scores"的新行，并初始化为0
        df.loc['scores'] = 0
        
        self.df = df
        
        self.t.append(0)
        
        # 创建随机分配的策略列表
        strategies = random.sample(self.col_index_10, self.s * self.N)
        self.strategies = strategies
        
        # 初始化所有人收益为0
        data_dict = {}
        for id, strategy1, strategy2 in zip(range(1, self.N+1), strategies[::2], strategies[1::2]):
            data_dict[(id, strategy1, strategy2)] = 0
        
        self.agentvalue = data_dict
        
        # 初始化所有人选择的策略为None
        tent_dict1 = {}
        tent_dict2 = {}
        for id in range(1, self.N+1):
           tent_dict1[id] = None
           tent_dict2[id] = None
        
        self.chosen_strategy = tent_dict1
        self.choices = tent_dict2
    
    # 更新市场状态
    def update_state(self,bit):
        new_string = self.memorystate[1:]
        new_string += str(bit)
        self.memorystate = new_string
    
    # 更新每个人的选择策略
    def choose_stategy(self):
        # 对字典中的每个人进行判断
        for key in self.agentvalue:
            id, strategy1, strategy2 = key
            
            # 获取两个策略的得分
            score1 = self.df.loc["scores",strategy1]
            score2 = self.df.loc["scores",strategy2]
            
            # 根据得分选择策略
            if score1 > score2:
                self.chosen_strategy[id] = strategy1
            
            elif score2 > score1:
                self.chosen_strategy[id] = strategy2
            else:
                self.chosen_strategy[id] = random.choice([strategy1, strategy2])
    
    # 更新每个人的选择
    def update_choices(self):
        for key in self.agentvalue:
            id, strategy1, strategy2 = key
            self.choices[id] = self.df.loc[self.memorystate,self.chosen_strategy[id]]
    
    # 更新获胜者
    def update_winchoice(self):
        count_0 = sum(value == 0 for value in self.choices.values())
        count_1 = sum(value == 1 for value in self.choices.values())
        if count_0 < count_1:
            self.winchoice = 0
        
        else:
            self.winchoice = 1
    
    # 更新策略价值和收益，进入下一轮
    def update_value(self):
        # 遍历数据集的每一列
        for column in self.df.columns:
            # 检查当前列中对应行索引的值是否等于给定值
            if column in self.chosen_strategy.values() and self.df.loc[self.memorystate, column] == self.winchoice:
                # 如果相等，则将对应列的"scores"加1
                self.df.loc["scores", column] += 1
                # 更新包含这些获胜策略的键的值加1
                for key in self.agentvalue:
                    id, strategy1, strategy2 = key
                    if column == strategy1 or column == strategy2:
                        self.agentvalue[key] += 1
        
        self.steps +=1
        self.t.append(self.steps)
        self.update_state(self.winchoice)
    
    # 循环
    def simulation(self):
        for i in range(self.iteration):
            self.choose_stategy()
            self.update_choices()
            self.update_winchoice()
            self.update_value()
            if i >= self.iteq:
                self.buyerlist.append(sum(self.choices.values()))
        
        self.var = np.var(np.array(self.buyerlist))

def get_varlist(numN):
    list_ = list()
    for i in range(8):
        game=Game(N=numN)
        game.init_game()
        game.simulation()
        list_.append(game.var)
    return list_

def total_simulation():
    average_sigma2_list = list()
    for numN in Nlist:
        varlist = get_varlist(numN)
        average = np.mean(varlist)
        std_deviation = np.std(varlist)
        average_sigma2_list.append(average)
    
    return average_sigma2_list

average_sigma2_list = total_simulation()

y = np.array(average_sigma2_list)/np.array(Nlist)


fig = plt.figure()

plt.plot(x, y, marker='o')

plt.title('σ^2/N versus 2^m/N')
plt.xlabel('2^m/N')
plt.ylabel('σ^2/N')

plt.show()

# The iteration number may not be large enough and there exists error due to randomness, but 2^m/N should be around 0.35~0.5 to 
# make the σ^2/N smallest.




