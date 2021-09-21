#Problem 1 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def e_field(theta):
    #values of z will be fed in as global variable
    R = 5  #chosen for this sphere
    ke = 8.99*10**8 #1/(4*pi*eps_0)                                                     
    sigma = 1 #pick 
    const = ke*2*np.pi*sigma*R**2
    int = const * np.sin(theta)*(z - R*np.cos(theta))/(R**2 + z**2 - 2*R*z*np.cos(theta))**(3/2)
    #pred = const/z**2*(1-(R-z)/np.abs(R-z))
    #err = int - pred
    return int

def integrator(fun,x0,x1,npt):
    x = np.linspace(x0,x1,npt)
    y = fun(x)
    dx = x1-x0
    area = dx/3*(y[0] + y[-1] + 2*np.sum(y[2:-1:2]) + 4*np.sum(y[1:-1:2])) #simpsons
    return area

npt = 30
x0 = 0
x1 = np.pi
z_vals = np.linspace(0,20,50)
E = np.empty(len(z_vals))
pred = np.empty(len(z_vals))
quad = np.empty(len(z_vals))
err = np.empty(len(z_vals))
for i,z in enumerate(z_vals):
    E[i] = integrator(e_field,x0,x1,npt) 
    #pred[i] = ke*2*np.pi*sigma*R**2/z**2*(1-(R-z)/np.abs(R-z))
    quad[i], err[i]  = integrate.quad(e_field,x0,x1)
 

plt.figure()
plt.plot(z_vals, E, label='integrator')
plt.plot(z_vals, quad, label='quad')  
plt.plot(5, 0, '.', label='R') #show a point for where R is
plt.xlabel('Distance z')
plt.ylabel('Electric field')
plt.legend()
plt.show()
#when z > R:
#E = ke * 4*np.pi*R**2*sigma/Z**2

#when z < R:
#E = 0
