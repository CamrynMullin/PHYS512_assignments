#problem 2 260926298
import numpy as np

x = np.linspace(1,2,10)
#x=42
def fun(x):
    return np.exp(x) #could be any other funciton

def ndiff(fun,x,full=False):
    eps = 2**-52 
    dx = eps**(1/3)
    f0, f1, f2 = fun(x), fun(x+dx), fun(x-dx)
    deriv = (f1-f2)/(2*dx)
    err = deriv/f0 - 1
 
    if full==True:
        return deriv, err
    return deriv

#print(ndiff(fun, x))
print(ndiff(fun, x, full=True))
