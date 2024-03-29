#problem 4 260926298
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#polynomial
def poly(x,y,npt1,npt2):
    x_fit=np.linspace(x[0],x[-1], npt2)
    y_fit=np.ones(len(x_fit))
    for i in range(1, npt1):
        y_fit*=(x_fit-x[i])
    y_fit = y_fit/y_fit[0]*y[0]
    return x_fit, y_fit 

#cubic spline
def spln(x,y,npt2):
    x_fit=np.linspace(x[0],x[-1],npt2)
    spln=interpolate.splrep(x,y)
    y_fit=interpolate.splev(x_fit, spln)
    return x_fit, y_fit

#rational funciton
def rat(x,y,npt1,npt2,n,m,pinv=False):
    assert(len(x)==n+m+1)
    top_mat = np.empty([npt1,n+1])
    bot_mat = np.empty([npt1,m])
    for i in range(n+1):
        top_mat[:,i] = x**i
    for i in range(m):
        bot_mat[:,i] = -y*x**(i+1)
    mat= np.hstack([top_mat, bot_mat])
    if pinv == False:
        params = np.linalg.inv(mat)@y
    else:
        params = np.linalg.pinv(mat)@y
    p = params[:n+1]
    q = params[n+1:]
    
    x_fit = np.linspace(x[0],x[-1],npt2)

    numer = 0
    for i,param in enumerate(p):
        numer += param*x_fit**i
    denom = 1
    for i, param in enumerate(q):
        denom += param*x_fit**(i+1)
    y_fit = numer/denom
    
    return x_fit, y_fit, p, q

npt1 = 10
npt2 = 2001

#cos(x)
print('For cos(x)')                                                                 
x = np.linspace(-np.pi/2, np.pi/2, npt1)
y = np.cos(x)
x_true = np.linspace(x[0],x[-1],npt2)
y_true = np.cos(x_true)

plt.figure()
#polynomial 
x_fit, y_fit = poly(x,y,npt1,npt2)
plt.plot(x_fit, y_fit, label = 'poly')
print('Error poly', y_fit-y_true)

#spline 
x_fit, y_fit = spln(x,y,npt2)
plt.plot(x_fit, y_fit, label = 'spline')
print('Error spline', y_fit-y_true)

#rational 
n=4
m=5
x_fit, y_fit, p, q = rat(x,y,npt1,npt2,n,m)
print('Error rat', y_fit-y_true)
plt.plot(x_fit, y_fit, label = 'rational')

plt.plot(x,y, '*', label = 'points')
plt.ylabel('cos(x)')
plt.legend()
plt.show()

print('--------------------')

#lorentzien
print('For Lorentzien')
x = np.linspace(-1,1, npt1)
y = 1/(1+x**2)
x_true = np.linspace(x[0],x[-1],npt2)
y_true = 1/(1+x_true**2)

plt.figure()
#polynomial
x_fit, y_fit = poly(x,y,npt1,npt2)
plt.plot(x_fit, y_fit, label = 'poly')
print('Error poly', np.abs(y_fit-y_true))

#spline
x_fit, y_fit = spln(x,y,npt2)
plt.plot(x_fit, y_fit, label = 'spline')
print('Error spline', np.abs(y_fit-y_true))

#rational
n=4
m=5
x_fit, y_fit, p, q = rat(x,y,npt1,npt2,n,m)
plt.plot(x_fit, y_fit, label = 'rational inv')
x_true = np.linspace(x[0],x[-1],npt2)
y_true = 1/(1+x_true**2)
pred_err = 2**(-52)*y_true*(-24*x_true**3+24*x_true)/(x_true**2+1)**4 #err = eps**(-52)*f(x)*f'''(x)
print('Predicted err', pred_err)
print('Error rat for np.linalg.inv', np.abs(y_fit-y_true))
print('p', p, 'q', q)
x_fit, y_fit, p, q = rat(x,y,npt1,npt2,n,m,pinv = True)
print('Error rat for np.linalg.pinv', np.abs(y_fit-y_true))
print('p', p, 'q', q)
plt.plot(x_fit, y_fit, label = 'rational pinv')

plt.plot(x,y, '*', label = 'points')
plt.ylabel('Lorentizan')
plt.legend()
plt.show()

