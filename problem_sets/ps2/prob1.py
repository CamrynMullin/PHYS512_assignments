#Problem 1 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def e_field(theta):
    #values of z will be fed in as global variable
    R = 1  #chosen for this sphere
    ke = 1/(4*np.pi) #1/(4*pi*eps) for eps = 1
    sigma = 1 #chosen 
    const = ke*2*np.pi*sigma*R**2
    int = const * np.sin(theta)*(z - R*np.cos(theta))/(R**2 + z**2 - 2*R*z*np.cos(theta))**(3/2)
    return int

def integrator(fun,x0,x1,tol):
    #use simpsons and adaptive step
    x = np.linspace(x0,x1,5)
    y = fun(x)
    dx = (x1-x0)/(len(x)-1)
    area_coarse = 2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area_fine = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err = np.abs(area_coarse - area_fine)
    if err<tol:
        return area_fine, err
    else:
        xmid=(x0+x1)/2
        left, errL = integrator(fun,x0,xmid,tol/2)
        right, errR = integrator(fun,xmid,x1,tol/2)
        return left+right, np.max([errL, errR])

npt = 10
#the theta values
x0 = 0
x1 = np.pi
z_vals = np.linspace(0, 20, 101)
E = np.empty(len(z_vals))
quad = np.empty(len(z_vals))
quad_err = np.empty(len(z_vals))
tol = 1e-7
for i,z in enumerate(z_vals):
    if z != 1.: #singluarity at z == R
        E[i], E_err = integrator(e_field,x0,x1,tol)
    else:
        E[i] = np.nan
    #integral using scipy's quad
    quad[i], quad_err[i]  = integrate.quad(e_field,x0,x1)
for i, e in enumerate(E): #filling in the missing value of E(x)
    if str(e) == 'nan':
        #take average of sourrounding values for gap in integral
        E[i] = (E[i-1] + E[i+1])/2
print('The max error for the integrator is', E_err,'and the max error for quad is', quad_err.max())

plt.figure()
plt.plot(z_vals, E, label='integrator')
plt.plot(z_vals, quad, '--',label='quad')  
plt.plot(1, 0, '.', label='R') #show a point for where R is
plt.xlabel('Distance z')
plt.ylabel('Electric field')
plt.legend()
plt.show()

