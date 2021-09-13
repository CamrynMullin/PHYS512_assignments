#problem 4 260926298
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

npt = 20
x = np.linspace(-np.pi/2, np.pi/2, npt)
y = np.cos(x)

#polynomial
def poly(x, y, npt):
    x_fit=np.linspace(x[0],x[-1],2001)
    p=np.empty(len(x_fit))
    for i in range(npt):
        p=(x_fit-x[i])
    p=p/p[0]*y[0]
    return x_fit, p 

#cubic spline
def spl(x,y,npt):
    x_fit=np.linspace(x[0],x[-1],2001)
    spln=interpolate.splrep(x,y)
    y_fit=interpolate.splev(x_fit, spln)
    return x_fit, y_fit
#rational funciton
#def rat(x,y,npt):
    


plt.figure()
plt.plot(x,y, '*')
plt.plot(poly(x,y,npt)[0], poly(x,y,npt)[1], label = 'poly')
plt.plot(spl(x,y,npt)[0], spl(x,y,npt)[1], label = 'spline')
plt.ylabel('cos(x)')
plt.legend()
plt.show()

x = np.linspace(-1,1, 100)
y = 1/(1+x**2)


plt.figure()
plt.plot(x,y, '*')
plt.plot(poly(x,y,npt)[0], poly(x,y,npt)[1], label = 'poly')
plt.plot(spl(x,y,npt)[0], spl(x,y,npt)[1], label = 'spline')
plt.ylabel('rational')
plt.legend()
plt.show()

