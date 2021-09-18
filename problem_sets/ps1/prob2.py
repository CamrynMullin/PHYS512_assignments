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
    deriv = (f1-f2)/(2*dx) #the derivative
    
    if full==True:
        #err = f(x)*eps/dx + 1/6*f'''(x)*dx**2
        deriv_3 = (fun(x+2*dx)-2*f1-fun(x-2*dx)+2*f2)/(2*dx**3)
        err = f0*eps/dx + 1/6*deriv_3*dx**2 #the error
        return deriv, dx, err
    return deriv

#print(ndiff(fun, x))
print('The derivative at the point(s) is',ndiff(fun, x, full=True)[0], 'with dx', ndiff(fun, x, full=True)[1], 'and with error', ndiff(fun, x, full=True)[2])

