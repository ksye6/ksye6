import pandas as pd
import numpy as np

obj = pd.Series(np.arange(5.), index=['a', 'c', 'c', 'd', 'e'])

#For series, it is pretty simple and straight forward
obj2 = obj.drop('c')
obj3 = obj.drop(['d', 'c'])

#With DataFrame, index values can be deleted from either axis. 

data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])

data1 = data.drop(['Colorado', 'Ohio'])

#You can drop values from the columns by passing axis=1 or axis='columns'
# data2 = data.drop('two', axis=1)
data2 = data.drop('two', axis='columns')

#Many functions, like drop, which modify the size or shape of a Series or 
#DataFrame, can manipulate an object in-place without returning a new object
print(obj)
obj.drop('c', inplace=True)
print(obj)

data3 = data2.copy()
### add new column
data3['five']=pd.NA
### add one row by loc
data3.loc['Hong Kong']=[16,17,18,pd.NA]



