#problem 1
import numpy as np

#a) found in pdf

#b)
x = 5
e = 2*10**(-52)
f1 = np.exp(x)
f2 = np.exp(0.01*x)
delta1 = (e*f1/f1)**(1/5)
delta2 = (e*f2/(0.01**5*f2))**(1/5)

err1 = e*f1/delta1**2 + f1*delta1**3
err2 = e*f2/delta2**2 + f2*delta2**3

print('error in exp(x)', err1, 'error of exp(0.01x)', err2)
