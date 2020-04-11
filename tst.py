from EDA.OutlierHandling import OutlierHandling
import inspect

def func2(args):
    return func1(args)

def func1(a):
    print(a)
    return a[0]+a[1]

def f2(*varargs):
    return func2(varargs)

# print(f2(1,2))
a=[1,2,3,4]

e=enumerate(a)
print(e.)