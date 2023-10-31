import pandas as pd
import numpy as np

# Example 1: series
S1 = pd.Series(np.linspace(11,19,9), index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],[1, 2, 3, 1, 3, 1, 2, 2, 3]])

S1['a']
S1.loc['a']
S1.loc['a',:]
S1.loc[['a','d'],1]
S1.loc[:,1]

A=S1.unstack()
B=A.stack()


# Example 2: dataframe
DF2 = pd.DataFrame(np.arange(12).reshape((4, 3)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])

# give names to index
DF2.index.names=['key1','key2']
DF2.columns.names=['state','color']

C=DF2.unstack()

# creat index
MI=pd.MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']], names=['state', 'color'])

# Reordering and Sorting Levels
DF2.swaplevel('key1', 'key2')
DF2.sort_index(level=1)
DF2.swaplevel(0, 1).sort_index(level=0)

# set_index() and reset_index()
D=DF2.reset_index('key1')
D2=D.reset_index('key2')
E=D2.set_index(['key1','key2'])
E.index
E.columns

