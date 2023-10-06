#1
import math
import pandas as pd

deg1=[]
sin1=[]
cos1=[]
tan1=[]
for i in range(0,13):
  deg1.append(30*i)
  sin1.append(format(round(math.sin(math.radians(30*i)),3)+0, ".3f"))
  cos1.append(format(round(math.cos(math.radians(30*i)),3)+0, ".3f"))
  if (i-3)%6==0:
    tan1.append("undef")
  else:
    tan1.append(format(round(math.tan(math.radians(30*i)),3)+0, ".3f"))

data={"deg":deg1,"sin":sin1,"cos":cos1,"tan":tan1}
df=pd.DataFrame(data)
print(df)

#2
import numpy as np
import random
maxnum=random.randint(3,5)
dim=random.randint(2,10)#You can easily use the input() to define the matrix dimension, this code is just for clearer display in Rmarkdown
#dim=input("Enter the dimension you want:")
matrix=[[random.randint(0,maxnum) for i in range(dim)] for j in range(dim)]
matrix1=np.array(matrix)
print(matrix1)

count=0
for i in range(0,dim):
  if matrix1[i][i]==0:
    count+=1
  if matrix1[i][dim-1-i]==0:
    count+=1

if dim%2!=0:
  if matrix1[int((dim-1)/2)][int((dim-1)/2)]==0:
    count-=1

print(count)

#3
def hollow_square(a,b):
  if isinstance(a,int) and a>=0 and isinstance(b,int) and b>=0 :
    for j in range(b):
      for i in range(a+2*b):
        print("#", end="")
      print()
    
    for j in range(a):
      for i in range(a+2*b):
        if i < b or i >= a+b:
          print("#", end="")
        else:
          print(" ", end="")
      print()
    
    for j in range(b):
      for i in range(a+2*b):
        print("#", end="")
      print()

  else:
    print("Your inputs need to be non-negative integers")

hollow_square(5,3)
hollow_square(6,2)
hollow_square(0,10)


def digital(a):
    if(a[0]=='-' and a[1:].isdigit() or a.isdigit()):
        return True
    else:
        return False

def hollow_square2():
  i=0
  while i==0:
    a=input("Enter the blank square's side length:")
    b=input("Enter the The distance from the blank square to the boundary:")
    if digital(a)==False or digital(b)==False:
      return
    if int(a)<0 or int(b)<0:
      print("Your inputs need to be non-negative integers",flush=True)###if flush=False then will be some mistakes.
    else:
      i=1
      
  a=int(a)
  b=int(b)
  
  if int(a)>=0 and int(b)>=0 :
    for j in range(b):
      for i in range(a+2*b):
        print("#", end="")
      print()
    
    for j in range(a):
      for i in range(a+2*b):
        if i < b or i >= a+b:
          print("#", end="")
        else:
          print(" ", end="")
      print()
    
    for j in range(b):
      for i in range(a+2*b):
        print("#", end="")
      print()
  return
