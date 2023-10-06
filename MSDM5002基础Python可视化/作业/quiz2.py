###### test here #######
title=input("Enter a title: ")


ban=["a","an","the","at","by","for","in","of","on","to","up","and","as","but","or","nor"]


words=title.split(" ")

tent=[]

for i in words:
  if i.lower() not in ban:
    tent.append(i.capitalize())
  else:
    tent.append(i)

result=" ".join(tent)
print(result)


# import re
# 
# j=re.split('[- ]', title)















###### function here #######
def mytitle(inpu):
  title=inpu
    
  ban=["a","an","the","at","by","for","in","of","on","to","up","and","as","but","or","nor"]
    
  words=title.split()
    
  tent=[]
  
  for i in words:
    if i.lower() not in ban:
      tent.append(i.capitalize())
    else:
      tent.append(i)
  
  result=" ".join(tent)
  return result


mytitle("Welcome to MSDM_5002 of data-driven modeling for MSc students offered by phys&math department in UST. We will continue to learn NumPy.")

