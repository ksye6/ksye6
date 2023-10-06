#1
import datetime
TODAY = str(datetime.date.today())

def has_space_find(s):
        if s.find(' ') == -1:
            return False
        else:
            return True


def my_copyright5(name='J. LIU',email='liuj@ust.hk',date=TODAY):
    len_name=len(name)
    str_len=len_name+27
    
    str_star='*'*(5+str_len+5)
    str_slash='***--'+'-'*str_len+'--***'
    
    print(str_star)
    print("***  Programmed by "+name+" for MSDM5002  ***")
    str_date="date: "+date
    print("***  "+" "*int((str_len-len(str_date))/2)+str_date+" "*int((str_len-len(str_date))/2+0.5)+"  ***")
    print(str_slash)
    
    
    statements='You can use it as you like, but there might be ' + \
        'many bugs. If you find some bugs, please send them to'
    statements=statements+' "'+email+'"'
    
    # ### you may improve this part
    start_point = 0
    end_point = start_point + str_len + 1
    k=0
    lines=0
    allchanges=0
    n_nochangelines=0
    while start_point < len(statements) and k==0:
      count=0
      if has_space_find(statements[start_point:end_point-1])==True and has_space_find(statements[start_point+str_len:end_point+str_len-1])==True:
        while statements[end_point-1-count]!=" ":
          count+=1
          allchanges+=1
        end_point=end_point-count
        print("***  " + statements[start_point:end_point] + count*" " +" ***")     
        start_point = end_point
        end_point = start_point + str_len + 1
        lines+=1
      else:
        print("***  " + statements[start_point:end_point-1] + " " +" ***")  
        n_nochangelines+=1
        start_point = end_point-1
        end_point = start_point + str_len + 1
        lines+=1
  
      if end_point>len(statements):
        k+=1
    
    print("***  " + statements[start_point:] + ((lines+1)*(str_len+1)-len(statements)-allchanges+1-n_nochangelines)*" "+ "***")   
    print(str_star)

my_copyright5()
my_copyright5('IA', 'ia@ust.hk')
my_copyright5('CHAN Tai Man', 'tmchan@asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf.com', '2023-01-01')

# 
# ##test
# 
# name='CHAN Tai Man'
# email='tmchan@asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf.com'
# len_name=len(name)
# str_len=len_name+27
# str_slash='***--'+'-'*str_len+'--***'
# statements='You can use it as you like, but there might be ' + \
#     'many bugs. If you find some bugs, please send them to'
# statements=statements+' "'+email+'"'
# 
# start_point = 0
# end_point = start_point + str_len + 1
# k=0
# lines=0
# allchanges=0
# n_nochangelines=0
# while start_point < len(statements) and k==0:
#   count=0
#   if has_space_find(statements[start_point:end_point-1])==True and has_space_find(statements[start_point+str_len:end_point+str_len-1])==True:
#     while statements[end_point-1-count]!=" ":
#       count+=1
#       allchanges+=1
#     end_point=end_point-count
#     print("***  " + statements[start_point:end_point] + count*" " +" ***")     
#     start_point = end_point
#     end_point = start_point + str_len + 1
#     lines+=1
#   else:
#     print("***  " + statements[start_point:end_point-1] + " " +" ***")  
#     n_nochangelines+=1
#     start_point = end_point-1
#     end_point = start_point + str_len + 1
#     lines+=1
#   
#   if end_point>len(statements):
#     k+=1
# 
# print("***  " + statements[start_point:] + ((lines+1)*(str_len+1)-len(statements)-allchanges+1-n_nochangelines)*" "+ "***")   
# print(str_slash)
# 

#2
import random
import pandas as pd

def simulate_gacha():
  listcard=["UR","SSR","SR","R"]
  # nUR=1
  # nSSR=5
  # nSR=20
  # nR=100
  listncard=[1,5,20,100]
  
  # pUR=0.001
  # pSSR=0.003
  # pSR=0.006
  # pR=0.00864
  
  listp=[0.001,0.003,0.006,0.00864]
  
  df=pd.DataFrame({"listcard":listcard,"listncard":listncard,"listp":listp})
  
  npull=0
  while True:
    npull+=1
    sump=0
    tentp=[]
    
    for i in range(4):
      sump+=df.loc[i,"listncard"]*df.loc[i,"listp"]
    
    for j in range(4):
      tentp.append(df.loc[j,"listncard"]*df.loc[j,"listp"]/sump)
    
    pull=random.choices(listcard,tentp)[0]
    df.loc[df["listcard"]==pull,"listncard"]-=1 #可选择放不放回，即有无保底
    
    if pull == 'UR':
      break
    
  return npull

simulate_gacha()

nsim=100
npull=0
for i in range(nsim):
  npull+=simulate_gacha()

average=npull/nsim
print(average)


#3
#(a)
#If one person walks in horizontal directions, his Y-axis is fixed; 
#if he can walks in vertical directions, his X-axis is fixed.

#(b)
import random

xlim=15
ylim=15

def is_upright(x,y):
  if (x==xlim) and (y==ylim):
    return True
  else:
    return False

def is_upleft(x,y):
  if (x==0) and (y==ylim):
    return True
  else:
    return False

def is_downright(x,y):
  if (x==xlim) and (y==0):
    return True
  else:
    return False

def is_downleft(x,y):
  if (x==0) and (y==0):
    return True
  else:
    return False


def is_crossroad(x,y):#十
  if (x//3>=1) and (y//3>=1) and (x%3==0) and (y%3==0) and (x!=xlim) and (y!=ylim):
    return True
  else:
    return False

def is_Trightjunction(x,y):#Tright
  if (x==xlim) and (y%3==0) and is_upright(x,y)==False and is_downright(x,y)==False:
    return True
  else:
    return False

def is_Tleftjunction(x,y):#Tleft
  if (x==0) and (y%3==0) and is_upleft(x,y)==False and is_downleft(x,y)==False:
    return True
  else:
    return False

def is_Tupjunction(x,y):#Tup
  if (x%3==0) and (y==ylim) and is_upright(x,y)==False and is_upleft(x,y)==False:
    return True
  else:
    return False

def is_Tdownjunction(x,y):#Tdown
  if (x%3==0) and (y==0) and is_downright(x,y)==False and is_downleft(x,y)==False:
    return True
  else:
    return False

def is_T_junction(x,y):#All T
  if is_Trightjunction(x,y)==True or is_Tleftjunction(x,y)==True or is_Tupjunction(x,y)==True or is_Tdownjunction(x,y)==True:
    return True
  else:
    return False

def is_horizontalcorridor(x,y):#水平
  if (y%3==0) and is_upright(x,y)==False and is_upleft(x,y)==False and is_downright(x,y)==False and is_downleft(x,y)==False and is_crossroad(x,y)==False and is_T_junction(x,y)==False:
    return True
  else:
    return False

def is_verticalcorridor(x,y):#垂直
  if (x%3==0) and is_upright(x,y)==False and is_upleft(x,y)==False and is_downright(x,y)==False and is_downleft(x,y)==False and is_crossroad(x,y)==False and is_T_junction(x,y)==False:
    return True
  else:
    return False


# check
# 
# count=0
# for i in range(16):
#   for j in range(16):
#     if XXX==True:
#       count+=1
# 
# print(count)

def choices(x,y):
  choices = []
  if is_upright(x,y):
    choices=["left","down"]

  elif is_upleft(x,y):
    choices=["right","down"]

  elif is_downright(x,y):
    choices=["left","up"]

  elif is_downleft(x,y):
    choices=["right","up"]

  elif is_crossroad(x,y):
    choices=["left","right","up","down"]

  elif is_Trightjunction(x,y):
    choices=["left","up","down"]

  elif is_Tleftjunction(x,y):
    choices=["right","up","down"]

  elif is_Tupjunction(x,y):
    choices=["left","right","down"]

  elif is_Tdownjunction(x,y):
    choices=["left","right","up"]

  elif is_horizontalcorridor(x,y):
    choices=["left","right"]

  elif is_verticalcorridor(x,y):
    choices=["up","down"]

  return choices


def step1(x,y):
  if x<0 or x>xlim or y<0 or y>ylim or (x%3!=0 and y%3!=0):
    return
  
  choice=choices(x,y)
  direction=random.choice(choice)
  
  if direction=="up":
    x1=x
    y1=y+1
  
  elif direction=="down":
    x1=x
    y1=y-1
  elif direction=="left":
    x1=x-1
    y1=y
  elif direction=="right":
    x1=x+1
    y1=y
  
  return x1,y1

#(c)
def simulate_supermarket(ax=15,ay=0,bx=0,by=15):
  stepa=0
  stepb=0
  
  while True:
    ax,ay=step1(ax,ay)
    stepa+=1
    if ax==bx and ay==by:
      return stepa+stepb
    
    bx,by=step1(bx,by)
    stepb+=1
    if bx==ax and by==ay:
      return stepa+stepb
    
  
random.seed(100)
result=0
nsim=4000
for i in range(nsim):
  result+=simulate_supermarket()

average=result/nsim
print(average)#784.2705

#(d)
def criterion(x1,y1,x2,y2):
  if y1==y2:
    if abs(x1-x2)<=3:
      if x1==0:
        x1interval=range(4)
      elif x1==xlim:
        x1interval=range(xlim-3,xlim+1)
      elif x1%3==0:
        x1interval=range(x1-3,x1+4)
      elif x1%3!=0:
        x1interval=range(x1//3*3,x1//3*3+4)
      
      if x2 in x1interval:
        return True
    
  if x1==x2:
    if abs(y1-y2)<=3:
      if y1==0:
        y1interval=range(4)
      elif y1==ylim:
        y1interval=range(ylim-3,ylim+1)
      elif y1%3==0:
        y1interval=range(y1-3,y1+4)
      elif y1%3!=0:
        y1interval=range(y1//3*3,y1//3*3+4)
      
      if y2 in y1interval:
        return True
  
  return False


def step2(x1,y1,x2,y2):
  if criterion(x1,y1,x2,y2):
    if x1==x2:
      if y1<y2:
        y1+=1
      elif y1>y2:
        y1-=1
    
    if y1==y2:
      if x1<x2:
        x1+=1
      elif x1>x2:
        x1-=1
    
  else:
    x1,y1=step1(x1,y1)
  
  return x1,y1


def simulate_supermarket_plus(ax=15,ay=0,bx=0,by=15):
  stepa=0
  stepb=0
  
  while True:
    ax,ay=step2(ax,ay,bx,by)
    stepa+=1
    if ax==bx and ay==by:
      return stepa+stepb
    
    bx,by=step2(bx,by,ax,ay)
    stepb+=1
    if bx==ax and by==ay:
      return stepa+stepb
    
  
random.seed(100)
result=0
nsim=4000
for i in range(nsim):
  result+=simulate_supermarket_plus()

average=result/nsim
print(average)#439.034


#4
import numpy as np
import time
import random
import math


num_pos=10
nbu=3
pos_take=[0,1,2,3,4]

##1

def func_check(num_test,pos_take):
  num_lose=0
  for nt in range(num_test):
    A=np.zeros(num_pos)
    
    #setup the first bullet
    B1=np.random.randint(0,num_pos)
    A[B1]=1
    
    #setup the second bullet
    B2=B1
    while B2==B1:
        B2=np.random.randint(0,num_pos)
    A[B2]=1
    
    #setup the third bullet
    B3=B1
    while B3==B1 or B3==B2:
        B3=np.random.randint(0,num_pos)
    A[B3]=1

    for n in range(num_pos):
        if A[n]==1:
            if n in pos_take:
                num_lose += 1
            break
    
  return(num_lose/num_test)

##2

def func_check2(num_test,nbu,pos_take):
  num_lose=0
  A=np.zeros(num_pos,int)
  A[pos_take]=1
  
  for nt in range(num_test):
    B1=min(random.sample(range(num_pos),nbu)) #np.random.choice(range(num_pos), size=nbu, replace=False, p=None)
    #types are different
    if A[B1]==1:
      num_lose+=1
    
  return(num_lose/num_test)


##3

n=10
b=3

def func_check3(n,b,pos_take):
  x=np.zeros(n,float)
  for k in range(0,n):
    if k+1>n-b+1:
      x[k]=0
    else:
      x[k]=math.factorial(n-b)*b*math.factorial(n-(k+1))/(math.factorial(n-b-(k+1)+1)*math.factorial(n))#easy to calculate
  
  return(sum(x[pos_take]))


##time test
pos_take_list=[pos_take,[9,8,7,6,5],[0,2,4,6,8]]

test_num=3
T=np.empty([3,test_num])

for i in range(len(pos_take_list)):
  for n in range(test_num):
    start_T=time.time(); func_check(100000,pos_take_list[i]);  T[0,n]=time.time()-start_T
    
    start_T=time.time(); func_check2(100000,3,pos_take_list[i]); T[1,n]=time.time()-start_T
    
    start_T=time.time(); func_check3(10,3,pos_take_list[i]); T[2,n]=time.time()-start_T

  test_time=T.sum(axis=1)/test_num
  print(test_time)


### I could just make the function 3 times faster(the func_check2), but I can use theory to rigorously calculate probabilities 
### like the func_check3. Why the order matters in the multiple bullet case is because the The probability distribution is skewed, 
### and the integral value in the first half of the function is larger, so the person is more likely to be shot.



