#problem 2 260926298
import numpy as np

def lorentz(x):
    return 1/(1+x**2)

def integrate_adaptive(fun,a,b,tol,extra=None):
    x = np.linspace(a,b,5)
    if extra is None:
        iter = 1
        y = fun(x)
    else:
        if extra[0] == a:
            iter = extra[-1]
            iter =+ 1
            y = np.concatenate(([extra[1]],fun(x[1:])))
        else:
            iter = extra[-1]
            iter += 1
            y = np.concatenate((fun(x[:5]),[extra[2]]))
    dx = (b-a)/(len(x)-1)
    area_coarse = 2*dx*(y[0]+4*y[2]+y[4])/3
    area_fine = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    err = np.abs(area_coarse - area_fine)
    if err<tol:
        #iter = 1
        return area_fine, iter
    else:
        mid = (a+b)/2
        left, iterL = integrate_adaptive(fun,a,mid,tol/2,extra=(a,y[0],y[4],iter))
        right, iterR  = integrate_adaptive(fun,mid,b,tol/2,extra=(b,y[0],y[4],iter))
        #print(iter)
        return left+right, iterL+iterR
a = 0
b = 1
tol = 1e-7
exp, iter  = integrate_adaptive(np.exp,a,b,tol)
#print(exp)
print('integral exp(x) from 0 to 1 is', exp, 'with',iter, 'function calls')

a = -100
b = 100
arctan, iter = integrate_adaptive(lorentz,a,b,tol)
#print(arctan)
print('integral lorentz(x) form -100 to 100', arctan, 'with', iter, 'function calls')

