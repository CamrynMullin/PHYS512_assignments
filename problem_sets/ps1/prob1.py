#problem 1 260926298
import numpy as np

#a) found in pdf

#b)
x = 42
eps = 2**(-52)
f1 = np.exp(x)
dx1 = (eps*f1/f1)**(1/5)
deriv1 = 1/(12*dx1)*(8*np.exp(x+dx1) - 8*np.exp(x-dx1) - np.exp(x+2*dx1) + np.exp(x-2*dx1))

print()
print('At x=42 the derivative of f(x)=exp(x) is', deriv1, 'with fractional error of', deriv1/f1-1)
print('The "real" derivative is', f1)

#x = 4200
f2 = np.exp(0.01*x)
dx2 = (eps*f2/(0.01**5*f2))**(1/5)
deriv2 = 1/(12*dx2)*(8*np.exp(0.01*(x+dx2)) - 8*np.exp(0.01*(x-dx2)) - np.exp(0.01*(x+2*dx2)) + np.exp(0.01*(x-2*dx2)))

print()
print('At x=42 the derivative of f(x)=exp(0.01x) is', deriv2, 'with fractional error of', deriv2/(0.01*f2)-1)
print('The "real" derivative is', 0.01*f2)
