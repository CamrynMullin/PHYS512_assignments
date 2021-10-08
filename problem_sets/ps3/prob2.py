#problem 2 Camryn Mullin 260926298
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt
#a)
#decay rates in seconds
hours = 60*60 #converting hours to seconds
years = 365*24*hours #converting years to seconds
U238 = 4.468e9*years
Th234 = 24.10*24*hours
Pr234 = 6.70*hours
U234 = 245500*years
Th230 = 75380*years
Ra226 = 1600*years
Rn222 = 3.82335*24*hours
Po218 = 3.10*hours
Pb214 = 26.8*hours
Bi214 = 19.9*hours
Po214 = 164.3e-6
Pb210 = 22.3*years
Bi210 = 5.015*years
Po210 = 138.376*24*hours
#Pb206 = 0 #the end
half_life = np.array([U238,Th234,Pr234,U234,Th230,Ra226,Rn222,Po218,Pb214,Bi214,Po214,Pb210,Bi210,Po210])

def decay_solver(x,y,half_life=half_life):
    dydx = np.zeros(len(half_life)+1)
    dydx[0] = -y[0]/half_life[0]
    for i in range((len(half_life)-1)):
        dydx[i+1] = y[i]/half_life[i] - y[i+1]/half_life[i+1]
    dydx[-1] = y[-2]/half_life[-1]
    return dydx#*np.log(2)

#t0,t1 are limits of integration, y0 is initial condition
t0 = 0
t1 = half_life[0]
y0 = np.zeros(len(half_life)+1)
y0[0] = 1
t_in = time.time()
rk4_sol = integrate.solve_ivp(decay_solver,[t0,t1],y0,method='Radau')
t_fin = time.time()
print('This took', rk4_sol.nfev,'evaluations and', t_fin-t_in,'seconds with integrate.solve_ivp method = Radau.')
print('The answers were', rk4_sol.y[:,-1], 'with truth', np.exp(-1*(t1-t0)))
#b)
a = 0#15
b = -1
x = np.linspace(rk4_sol.t[a], rk4_sol.t[b-1], len(rk4_sol.t[a:b]))
y_U238 = rk4_sol.y[0,a:b]
y_Pb206 = rk4_sol.y[-1,a:b] #Pb210
plt.figure()
plt.plot(x, y_Pb206/y_U238)
plt.xlabel('Time')
plt.ylabel('Pb206/U238')
plt.title('Ratio of Pb206 to U238')
plt.savefig('Ratio of Pb206 to U238.png')
plt.show()

a = 0#10
b = -1#25
x = np.linspace(rk4_sol.t[a], rk4_sol.t[b-1], len(rk4_sol.t[a:b]))
y_Th230 = rk4_sol.y[4,a:b]
y_U234 = rk4_sol.y[3,a:b] 
plt.figure()
plt.plot(x, y_Th230/y_U234)
plt.xlabel('Time')
plt.ylabel('Th230/U234')
plt.title('Ratio of Th230 to U234')
plt.savefig('Ratio of Th230 to U234.png')
plt.show()