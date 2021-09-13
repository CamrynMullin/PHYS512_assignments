#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

dat = np.loadtxt('lakeshore.txt')
data = dat.T
V = np.linspace(data[1][0], data[1][-1], 10)
#V = data[1][60]

def lakeshore(V,data):
    T_pts = data[0] #tempurature
    V_pts = data[1] #voltage
    dV_dT = data[2] #dV/dT    
    
    T_line = np.linspace(data[0][0], data[0][-1], 1000)
    spln = interpolate.splrep(T_pts, V_pts)
    V_line = interpolate.splev(T_line, spln)

    try: 
        num = len(V)
    except TypeError:
        V = [V]
        num = len(V)
    T = np.empty(num)
    for i in range(num):
        for j, volt in enumerate(V_line):
            if np.round(V[i], 3) == np.round(volt,3):
                T[i] = T_line[j]
    err = 'error' 
    return T, err

print('At the voltage', V, 'the tempurature is', lakeshore(V,data)[0], 'with uncertainty', lakeshore(V,data)[1])
plt.plot(data[0],data[1],  label='Temp vs Volt relation')
plt.plot(lakeshore(V,data)[0],V, '*',  label='interpolated')
#plt.plot(lakeshore(V,data)[2], lakeshore(V, data)[3],  label='temp points')
plt.ylabel('Voltage')
plt.xlabel('Temp')
plt.legend()
plt.show()
