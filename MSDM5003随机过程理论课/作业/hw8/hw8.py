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
        self.t=list() # �����б�
        self.memorystate = format(0, '05b')  # ��ʼ�г�״̬
        self.strategies = None # ���в�������
        self.chosen_strategy = None # ������ÿ��ѡ��Ĳ�������
        self.choices = None # ÿ���˵�ѡ��0����1��
        self.agentvalue = None # ÿ���˵�����
        self.df = None # ��ϸ��������
        self.winchoice = None # ������
        self.buyerlist = list() #1
        self.var = None
    
    def init_game(self):
        # ������
        self.row_index = [format(i, '05b') for i in range(self.nrow)]
        
        # ����������
        self.col_index_10 = random.sample(range(2**self.nrow), self.N*self.s)
        self.col_index_2 = [format(i, '032b') for i in self.col_index_10]
        
        # �������ݼ�
        data = [[int(col[i]) for col in self.col_index_2] for i in range(self.nrow)]
        
        # ����DataFrame
        df = pd.DataFrame(data, index=self.row_index, columns=self.col_index_10)
        
        # ������Ϊ"scores"�����У�����ʼ��Ϊ0
        df.loc['scores'] = 0
        
        self.df = df
        
        self.t.append(0)
        
        # �����������Ĳ����б�
        strategies = random.sample(self.col_index_10, self.s * self.N)
        self.strategies = strategies
        
        # ��ʼ������������Ϊ0
        data_dict = {}
        for id, strategy1, strategy2 in zip(range(1, self.N+1), strategies[::2], strategies[1::2]):
            data_dict[(id, strategy1, strategy2)] = 0
        
        self.agentvalue = data_dict
        
        # ��ʼ��������ѡ��Ĳ���ΪNone
        tent_dict1 = {}
        tent_dict2 = {}
        for id in range(1, self.N+1):
           tent_dict1[id] = None
           tent_dict2[id] = None
        
        self.chosen_strategy = tent_dict1
        self.choices = tent_dict2
    
    # �����г�״̬
    def update_state(self,bit):
        new_string = self.memorystate[1:]
        new_string += str(bit)
        self.memorystate = new_string
    
    # ����ÿ���˵�ѡ�����
    def choose_stategy(self):
        # ���ֵ��е�ÿ���˽����ж�
        for key in self.agentvalue:
            id, strategy1, strategy2 = key
            
            # ��ȡ�������Եĵ÷�
            score1 = self.df.loc["scores",strategy1]
            score2 = self.df.loc["scores",strategy2]
            
            # ���ݵ÷�ѡ�����
            if score1 > score2:
                self.chosen_strategy[id] = strategy1
            
            elif score2 > score1:
                self.chosen_strategy[id] = strategy2
            else:
                self.chosen_strategy[id] = random.choice([strategy1, strategy2])
    
    # ����ÿ���˵�ѡ��
    def update_choices(self):
        for key in self.agentvalue:
            id, strategy1, strategy2 = key
            self.choices[id] = self.df.loc[self.memorystate,self.chosen_strategy[id]]
    
    # ���»�ʤ��
    def update_winchoice(self):
        count_0 = sum(value == 0 for value in self.choices.values())
        count_1 = sum(value == 1 for value in self.choices.values())
        if count_0 < count_1:
            self.winchoice = 0
        
        else:
            self.winchoice = 1
    
    # ���²��Լ�ֵ�����棬������һ��
    def update_value(self):
        # �������ݼ���ÿһ��
        for column in self.df.columns:
            # ��鵱ǰ���ж�Ӧ��������ֵ�Ƿ���ڸ���ֵ
            if column in self.chosen_strategy.values() and self.df.loc[self.memorystate, column] == self.winchoice:
                # �����ȣ��򽫶�Ӧ�е�"scores"��1
                self.df.loc["scores", column] += 1
                # ���°�����Щ��ʤ���Եļ���ֵ��1
                for key in self.agentvalue:
                    id, strategy1, strategy2 = key
                    if column == strategy1 or column == strategy2:
                        self.agentvalue[key] += 1
        
        self.steps +=1
        self.t.append(self.steps)
        self.update_state(self.winchoice)
    
    # ѭ��
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

plt.title('��^2/N versus 2^m/N')
plt.xlabel('2^m/N')
plt.ylabel('��^2/N')

plt.show()

# The iteration number may not be large enough and there exists error due to randomness, but 2^m/N should be around 0.35~0.5 to 
# make the ��^2/N smallest.




