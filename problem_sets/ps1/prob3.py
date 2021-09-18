#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

dat = np.loadtxt('lakeshore.txt')
data = dat.T
V = np.linspace(data[1][0], data[1][-1], 10) #V an array
#V = 0.8 #single V

def lakeshore(V,data):
    T_pts = data[0] #tempurature
    V_pts = data[1] #voltage    
    
    V_use, T_use= zip(*sorted(zip(V_pts, T_pts))) #need to sort x values
    spln = interpolate.splrep(V_use,T_use)
    T = interpolate.splev(V,spln)
    
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
        var.append(np.var(spln[0])) #variance in spline at each point
    err = np.var(var)
    return T, err

print('At the voltage', V, 'the tempurature is', lakeshore(V,data)[0], 'with uncertainty', lakeshore(V,data)[1])
plt.plot(data[1],data[0],  label='Temp vs Volt relation')
plt.plot(V, lakeshore(V,data)[0], '*',  label='interpolated')
plt.xlabel('Voltage (V)')
plt.ylabel('Temp (k)')
plt.legend()
plt.show()
