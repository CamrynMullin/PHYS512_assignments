#problem 3
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('dish_zenith.txt')
xyz = data.T

def fun(params, x, y):
    return (x-min(np.abs(x)))*params[0] + (y-min(np.abs(y)))*params[1] + params[2]
    #return (x-np.min(np.abs(x)))**2 + (y-np.min(np.abs(y)))

x = xyz[0]
y = xyz[1]
z = xyz[2]
a = 0.001
params = np.array([2*a*min(np.abs(x)), 2*a*min(np.abs(y)), min(z)])
z_fit = fun(params,x,y)
nd = len(x)
nm = len(xyz)
A = np.zeros([nd, nm])
A[:,0] = x
A[:,1] = y 
A[:,2] = 1
#assume noise is 1 for now
lhs = A.T@A
rhs = A.T@z_fit
m = np.linalg.pinv(lhs)@rhs
#u,s,v=np.linalg.svd(A,0)
#m = v@u@s@z
z_pred = A@m
chi2 = np.sum((z_fit-z_pred)**2)
print('The residual is', chi2)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot3D(x, y, z_fit, 'blue', label = 'fit')
ax.plot3D(x, y, z, 'red', label = 'real')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
#ax.plot3D(np.linspace(min(x), max(x), len(x)), np.linspace(min(y), max(y), len(y)), z_fit - z_pred)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.title('Residuals')
#plt.show()

#noise != 1
noise = np.std(z_fit - z_pred)
print('I think the per-point noise is', noise)

N = np.eye(len(x))*noise**2
mat = A.T@np.linalg.inv(N)@A
errs = np.linalg.inv(mat)
print('parameter errors are', np.sqrt(np.diag(errs)))

derrs = A@errs@A.T
model_sig = np.sqrt(np.diag(derrs))
