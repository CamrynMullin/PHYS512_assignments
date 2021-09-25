#problem 2 260926298
import numpy as np

def lorentz(x):
    return 1/(1+x**2)

def integrate_adaptive(fun,a,b,tol,extra=None):
    global counter
    x = np.linspace(a,b,5)
    if extra is None:
        y = fun(x)
        counter += len(y)
    else:
        y = np.empty(5)
        for i,xi in enumerate(x):
            if xi in extra[0]:
                j = np.where(extra[0] == xi)
                y[i] = extra[1][j]
            else:
                y[i] = fun(xi)
                counter+=1
    dx = (b-a)/(len(x)-1)
    area_coarse = 2*dx*(y[0]+4*y[2]+y[4])/3
    area_fine = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    err = np.abs(area_coarse - area_fine)
    if err<tol:
        return area_fine
    else:
        mid = (a+b)/2
        left = integrate_adaptive(fun,a,mid,tol/2,extra=(x,y))
        right = integrate_adaptive(fun,mid,b,tol/2,extra=(x,y))
        return left+right
a = 0
b = 1
tol = 1e-7
counter = 0
exp = integrate_adaptive(np.exp,a,b,tol)
err = np.abs(exp-(np.exp(b) - np.exp(a)))
print('integral exp(x) from 0 to 1 is', exp, 'with error', err, 'and with',counter, 'function calls. This saved', 135-counter, 'calls compaired to the "lazy" method in class.' )

a = -100
b = 100
counter = 0
arctan = integrate_adaptive(lorentz,a,b,tol)
err = np.abs(arctan-(np.arctan(b) - np.arctan(a)))
print('integral lorentz(x) form -100 to 100', arctan, 'with error', err,'and with', counter, 'function calls. This saved', 5175-counter, 'calls compared to the "lazy" method used in class.')

