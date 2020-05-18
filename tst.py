from EDA.OutlierHandling import OutlierHandling
import inspect

def func2(args):
    return func1(args)

def func1(a):
    print(a)
    return a[0]+a[1]

def f2(*varargs):
    return func2(varargs)


from itertools import combinations
from time import time
import random
pos=[]
negative=[]
for i in range(20):
    pos.append(random.randint(1,1000))
    negative.append(random.randint(-1000, -1))
count=0
startTime=time()
for i in range(1, len(pos)+1):
    com=combinations(pos, i)
    for c in list(com):
        s=sum(list(c))
        neg=list(filter(lambda x: (-x)<=s, negative))
        for j in range(1, len(neg)+1):
            negCom=list(combinations(neg, j))
            for nc in negCom:
                if s+sum(list(nc))==0:
                    count+=1
endTime=time()
print('#combinations: {}'.format(count))
print('Time taken {}'.format(endTime-startTime))