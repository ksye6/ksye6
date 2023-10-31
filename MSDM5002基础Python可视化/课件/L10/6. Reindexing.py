import pandas as pd
import numpy as np

obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'a', 'b', 'c', 'd', 'e'])

obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj4 = obj3.reindex(range(6), method='ffill')


#With DataFrame, reindex can alter either the (row) index, columns, or both.
frame1 = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'], columns=['Ohio', 'Utah', 'Texas'])
frame2 = frame1.reindex(['a', 'b', 'c', 'd'])

#The columns can be reindexed with the columns keyword
states = ['Utah', 'Texas']
frame3 = frame1.reindex(columns=states)

#you can reindex more succinctly by label-indexing with loc
frame4 = frame1.loc[['a', 'c', 'd'], states]

# ##Q1. Can use reindex the non-exsiting column
# states = ['Texas', 'Utah', 'California']
# frame3 = frame1.reindex(columns=states)

# ##Q2. Can we use loc to reindex the non-existing rows?
# states = ['Utah', 'Texas']
# frame4 = frame1.loc[['a', 'b', 'c', 'd'], states]


################# for multi-level index
#you can use set_index and reset_index to 
#realize multilevel idnex
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame0 = pd.DataFrame(data)

frame5=frame0.set_index(['state','year'])
frame0_back=frame5.reset_index(level=[0,1])
frame0_back1=frame5.reset_index('state')
frame0_back2=frame0_back1.reset_index('year')

#different to access the multiple level index
print(frame5.loc['Ohio'].loc[2001])
print(frame5.loc['Ohio',2001])
print(frame5.iloc[2])


