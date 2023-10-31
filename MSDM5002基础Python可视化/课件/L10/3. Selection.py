import pandas as pd
import numpy as np

#Series indexing (obj[...]) works analogously to NumPy array indexing
#except you can use the Series’s index values instead of only integers

obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print(obj['b']); print(obj[1]);
print(obj[['b', 'a', 'd']])
print(obj[[1, 3]])
print(obj[obj < 2])
#Slicing with labels behaves differently than normal Python slicing in that 
#the end‐point is inclusive
print(obj['b':'c'])

#Setting using these methods modifies the corresponding section of the Series
obj['b':'c'] = 5; print(obj)

#Indexing into a DataFrame is for retrieving one or more columns either with 
#a single value or sequence
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
print(data['two']); print(data[['three', 'one']])
print(data[data['three'] > 5])

#change the selected elements
data[data < 5] = 0
print(data)

#loc and iloc enable you to select a subset of the rows and columns from a DataFrame
#using either axis labels (loc) or integers (iloc)
#Both indexing functions work with slices in addition to single labels or lists of labels
print(data.loc['Colorado', ['two', 'three']]); print(data.loc[:'Utah', 'two'])
print(data.iloc[1, [1,2]]); print(data.iloc[[1,3], [1,2]])

#Both indexing functions work with slices in addition to single labels or lists of labels
print(data.loc[:'Utah', 'two'])
print(data.iloc[:, :3][data.three > 5])








