#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors

dat = np.loadtxt('lakeshore.txt')
data = dat.T
V = np.linspace(data[1][0], data[1][-1], 10)
#V = data[1][60]

def lakeshore(V,data):
    T_pts = data[0] #tempurature
    V_pts = data[1] #voltage
    dV_dT = data[2] #dV/dT    
    
    try:
        npts = len(V)
    except TypeError:
        V = np.array([V])
        npts = len(V)

    V_pts2 = V_pts.reshape(-1,1)
    V2 = V.reshape(-1,1)
    nbrs  = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(V_pts2)
    distances, indicies  = nbrs.kneighbors(V2, 4)
    V_interp = np.empty(np.shape(indicies))
    T_interp = np.empty(np.shape(indicies))
    for i in range(npts):
        for j in range(len(indicies[0])): 
            V_interp[i][j] = V_pts[indicies[i,j]]
            T_interp[i][j] = T_pts[indicies[i,j]]
    #print(V_interp)
    T = np.empty(npts)
    for i in range(npts):
        poly = np.polyfit(V_interp[i],T_interp[i],3)
        T[i] = np.polyval(poly, V[i])
    err = 'error' 
    return T, err

print('At the voltage', V, 'the tempurature is', lakeshore(V,data)[0], 'with uncertainty', lakeshore(V,data)[1])
plt.plot(data[0],data[1],  label='Temp vs Volt relation')
plt.plot(lakeshore(V,data)[0],V, '*',  label='interpolated')
#plt.plot(lakeshore(V,data)[2], lakeshore(V, data)[3],  label='temp points')
plt.ylabel('Voltage (V)')
plt.xlabel('Temp (k)')
plt.legend()
plt.show()
