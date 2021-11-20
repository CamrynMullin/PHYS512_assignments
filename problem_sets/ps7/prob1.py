#problem set 7 prob 1Camryn Mullin 260926298
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import ctypes
import numba as nb

#random number generator from C
points = np.loadtxt('rand_points.txt')
x = points.T[0]
y = points.T[1]
z = points.T[2]

#plotting the data from C
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x,y,z, '.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('C_random.png')
plt.show()

#from this plot it can be seen that the points are distributed in rows and columns

#testing python
x_py = np.random.randint(10**8,size = len(x))
y_py = np.random.randint(10**8,size = len(x))
z_py = np.random.randint(10**8,size = len(x))

#plotting the data from python
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_py,y_py,z_py, '.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('Python_random.png')
plt.show()

#the numpy random number generatior does not seem to suffer from this same problem

#My computer random number generator
mylib = ctypes.cdll.LoadLibrary("libc.dylib")
rand = mylib.rand
rand.argtypes = []
rand.restype = ctypes.c_int

#from sample code
@nb.njit
def get_rands_nb(vals):
    n = len(vals)
    for i in range(n):
        vals[i] = rand()
    return vals

def get_rands(n):
    vec = np.empty(n,dtype='int32')
    get_rands_nb(vec)
    return vec

n = 300000000
vec = get_rands(n*3)                                                            

vv = np.reshape(vec,[n,3])
vmax = np.max(vv,axis=1)

maxval = 1e8
vv2 = vv[vmax<maxval,:]

x_mine = vv2.T[0]
y_mine = vv2.T[1]
z_mine = vv2.T[2]

#plotting the data from C using my computer
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x,y,z, '.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('mine_random.png')
plt.show()

#the numbers from the C library run on my machine seem to give the same issue
