#problem 2 Camryn Mullin 260926298
import numpy as np
from scipy import integrate
import time

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
#Pb206 the end
half_life = np.array([U238,Th234,Pr234,U234,Th230,Ra226,Rn222,Po218,Pb214,Bi214,Po214,Pb210,Bi210,Po210])

def decay_solver(x,y,half_life=half_life):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    for i in range((len(half_life[1:-1]))):
        dydx[i] = y[i-1]/half_life[i-1]-y[i]/half_life[i]
    dydx[-1]=y[-1]/half_life[-1]
    return dydx    

#t0,t1 are limits of integration, y0 is initial condition
t0 = 0
t1 = np.sum(half_life)
y0 = np.zeros(len(half_life)+1)
y0[0] = 1
t_in = time.time()
rk4_sol = integrate.solve_ivp(decay_solver,[t0,t1],y0,method='Radau')
t_fin = time.time()
print('This took', rk4_sol.nfev,'evaluations and', t_fin-t_in,'seconds with integrate.solve_ivp method = Radau.')
print('The answers were', rk4_sol.y[0,-1], 'with truth', np.exp(-1*(t1-t0)))