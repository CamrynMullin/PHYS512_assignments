#problem 2 260926298
import numpy as np

def lorentz(x):
    return 1/(1+x**2)

def integrate_adaptive(fun,a,b,tol,extra=None):
    global counter
    if extra is None:
        x = np.linspace(a,b,5)
        y = fun(x)
        counter += len(y)
    else:
        x = np.linspace(a,b,5)
        if extra[0] == a:
            y1 = fun(x[1:])
            y = np.concatenate(([extra[1]],y1))
        else:
            y1 = fun(x[:5])
            y = np.concatenate((y1,[extra[1]]))
        counter += len(y1)
    dx = (b-a)/(len(x)-1)
    area_coarse = 2*dx*(y[0]+4*y[2]+y[4])/3
    area_fine = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    err = np.abs(area_coarse - area_fine)
    if err<tol:
        return area_fine
    else:
        mid = (a+b)/2
        left = integrate_adaptive(fun,a,mid,tol/2,extra=(a,y[0]))
        right = integrate_adaptive(fun,mid,b,tol/2,extra=(b,y[4]))
        return left+right
a = 0
b = 1
tol = 1e-7
counter = 0
exp = integrate_adaptive(np.exp,a,b,tol)
#print(exp)
print('integral exp(x) from 0 to 1 is', exp, 'with',counter, 'function calls. In class this took 135 function calls, meaning we saved', 135-counter, 'calls.' )

a = -100
b = 100
counter = 0
arctan = integrate_adaptive(lorentz,a,b,tol)
#print(arctan)
print('integral lorentz(x) form -100 to 100', arctan, 'with', counter, 'function calls. In class this took 5175 function calls, meaning we saved', 5175-counter, 'calls.')

