#problem 3 Camryn Mullin 260926298
import numpy as np
import matplotlib.pyplot as plt

def fun(params, x, y):
    return params[0]*(x**2 + y**2) - params[1]*x - params[2]*y + params[3]

data = np.loadtxt('dish_zenith.txt')
xyz = data.T
x = xyz[0]
y = xyz[1]
z = xyz[2]

#part b
nd = len(x)
nm = 4
A = np.zeros([nd, nm])
A[:,0] = x**2 + y**2
A[:,1] = x
A[:,2] = y
A[:,3] = 1
#assume noise is 1 for now
lhs = A.T@A
rhs = A.T@z
m = np.linalg.inv(lhs)@rhs
z_pred = A@m

print('The best fit parameters a,b,c,d are', m)
a = m[0]
x0 = m[1]/(m[0]*2*-1)
y0 = m[2]/(m[0]*2*-1)
z0 = m[3] - a*x0**2 - a*y0**2
print('a=', a, 'x0=', x0, 'y0=', y0,'z0=', z0)

z_fit = fun(m,x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot3D(x, y, z_fit, 'blue', label = 'fit')
ax.plot3D(x, y, z_pred, 'red', label = 'prediced')
ax.plot3D(x, y, z, 'green', label = 'real data')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.savefig('Telescope with N=1.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot3D(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)), z_fit-z_pred, '.')
ax.plot3D(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)), np.zeros(len(z)))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('Residuals.png')
plt.show()

#part c
N = np.eye(len(x))*(z_fit - z_pred)**2 #noise is diagonal matrix of errors squared
N_inv = np.linalg.inv(N)
lhs_new = A.T@N_inv@A
rhs_new = A.T@N_inv@z
m_new = np.linalg.inv(lhs_new)@rhs_new
z_new = fun(m_new, x,y)

errs = np.linalg.inv(lhs_new)
param_errs = np.sqrt(np.diag(errs)) #find uncertainty in a
print('The uncertaintly in a is', param_errs[0])

f = np.abs(z_new.min()) #focal length
print('The focal length in meters is ', f/1000)
i = np.where(np.abs(z_new) == f)
z_errs=A@errs@A.T
model_sigma=np.sqrt(np.diag(z_errs))
print('The error of focal length is +-', float(model_sigma[i])/1000, 'm')
