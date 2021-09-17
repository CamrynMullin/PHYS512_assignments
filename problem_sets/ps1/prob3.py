#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

dat = np.loadtxt('lakeshore.txt')
data = dat.T
V = np.linspace(data[1][0], data[1][-1], 10)
#V = 0.8

def lakeshore(V,data):
    T_pts = data[0] #tempurature
    V_pts = data[1] #voltage
    dV_dT = data[2] #dV/dT    
    
    #try:
        #npt = len(V)
    #except TypeError:
        #V = np.array([V])
        #npt = len(V)
    
    #V_pts2 = V_pts.reshape(-1,1)
    #V2 = V.reshape(-1,1)
    #nbrs  = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(V_pts2)
    #distances, indicies  = nbrs.kneighbors(V2, 4)
    #V_use = V_pts[indicies]
    #T_use = T_pts[indicies] 
    V_use, T_use= zip(*sorted(zip(V_pts, T_pts)))
    
    spln=interpolate.splrep(V_use,T_use)
    #print((spln))
    T=interpolate.splev(V,spln)
    #T = np.empty(npt)
    #for i in range(npt):
        #poly = np.polyfit(V_use[i],T_use[i],3)
        #T[i] = np.polyval(poly, V[i])
    
    #the error
    bootstrap1 = np.random.choice(np.linspace(0, len(V_pts)-1, len(V_pts), dtype=int), len(V_pts)) #bootstrap sample of indicies
    bootstrap2 = np.random.choice(np.linspace(0, len(V_pts)-1, len(V_pts), dtype=int), len(V_pts))
    bootstrap3 = np.random.choice(np.linspace(0, len(V_pts)-1, len(V_pts), dtype=int), len(V_pts))
    bootstrap_set = [bootstrap1, bootstrap2, bootstrap3]
    
    var = []
    for bootstrap in bootstrap_set:      
        V_boot = V_pts[bootstrap]
        T_boot = T_pts[bootstrap]
        V_use, T_use= zip(*sorted(zip(V_boot, T_boot)))
        spln = interpolate.splrep(V_use,T_use)
        var.append(np.var(spln[0]))
    err = np.var(var)
    #err = 'e'
    return T, err

print('At the voltage', V, 'the tempurature is', lakeshore(V,data)[0], 'with uncertainty', lakeshore(V,data)[1])
plt.plot(data[1],data[0],  label='Temp vs Volt relation')
plt.plot(V, lakeshore(V,data)[0], '*',  label='interpolated')
#plt.plot(lakeshore(V,data)[2], lakeshore(V, data)[3],  label='temp points')
plt.xlabel('Voltage (V)')
plt.ylabel('Temp (k)')
plt.legend()
plt.show()
