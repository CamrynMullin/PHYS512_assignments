#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt('lakeshore.txt')
data = dat.T
V = np.linspace(data[1][2], data[1][-3], len(data[0]))
#V = data[1][60]

def lakeshore(V,data):
    T_true = data[0] #tempurature
    V_true = data[1] #voltage
    dV_dT = data[2]    
    try:
        T = np.empty(len(V))
    except TypeError:
        V = [V]
        T = np.empty(len(V))
    for i in range(len(V)):
        if i ==0 : 
            dV = np.abs(V_true[i+1] - V_true[i])
        else:
            dV = np.abs(V_true[i] - V_true[i-1])
        ind = np.abs(V[i]- V_true[0])/dV
        ind = int(np.floor(ind))
        x = V_true[ind-1:ind+3]
        y = T_true[ind-1:ind+3]
        p = np.polyfit(x, y, 3)
        T[i] = np.polyval(p, V[i])
    return T
print('Voltage at a tempurature of', V, 'is', lakeshore(V,data))
plt.plot(data[1],data[0]), label='true fit')
plt.plot(V, lakeshore(V,data), '*', label='interpolated')
plt.xlabel('Voltage')
plt.ylabel('Temp')
plt.legend()
plt.show()
