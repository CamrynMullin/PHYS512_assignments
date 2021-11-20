#prob 3 ps7 Camryn Mullin 260926298
import numpy as  np
import matplotlib.pyplot as plt
from scipy.special import erf

def make_hist(x,bins):
    hist, bin_edges = np.histogram(x,bins) #x values, edges
    centers = 0.5*(bin_edges[1:] + bin_edges[:-1]) #wanna compare at center
    hist = hist/hist.sum()
    return hist, centers #normalized histogram and centers

# exponential: x = e^(-t) = CDF^-1(q), CDF: e^(-t) = q --> t = -ln(q)
n = 100000
u = np.linspace(0,1,n+1)
u = u[1:] #ln(0) is problematic
#have u < sqrt(ae^-ar)
# v = -u/a * ln(u^2/a)
# let a = 1
v = -u*np.log(u**2)

plt.figure()
plt.plot(u, v,'b', label = 'v')
plt.plot(u, -v,'r', label = '-v')
plt.xlim(0,1.5)
plt.title('Bounding from ratio of uniforms')
plt.legend()
#plt.savefig('lorentzian_vs_exp_curve.png')
plt.show()

#ratio-of-uniforms
u = np.random.rand(n)
v = (2*np.random.rand(n) - 1)*0.8
r = v/u
# exponential: x = e^(-t)
accept = u < np.sqrt(np.exp(-r)) #accept y < sqrt(p(v/u))
t_use = r[accept]

bins = np.linspace(1,20,101)
hist, centers = make_hist(t_use,bins)
pred_exp = np.exp(-centers) #PDF
pred_exp = pred_exp/pred_exp.sum()

plt.figure()
plt.ion()
plt.plot(centers, hist, '*', label = 'deviates')
plt.plot(centers, pred_exp, 'r', label = 'exponential')
plt.title('Exponential Deviates from Ratio of Uniforms')
plt.legend()
plt.savefig('q3_exp_deviates.png')
plt.show()

def eff(y, accept):
    used = y[accept]
    return len(used)/len(y) * 100
efficiency = eff(r, accept)
print('the efficiency is', efficiency, '%')

