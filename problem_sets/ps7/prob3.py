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
u = np.linspace(0,1,2001)
u = u[1:]
#have u < sqrt(e^-r)
#ln(u) = 0.5*-r = 0.5*-v/u
# v = -ln(u)*u*2
v = -2*u*np.log(u)
print('max v is', v.max())

plt.figure()
plt.plot(u,v,'k')
plt.plot(u,-v,'k')
plt.show()

#ratio-of-uniforms
n = 100000
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
plt.legend()
plt.savefig('exp_ratio.png')
plt.show()


##check the output
#print('mean and std are ',t_use.mean(),np.std(t_use))
#print('2 and 3-sigma fractions are ',np.mean(np.abs(t)<2),np.mean(np.abs(t)<3))
##print('expected are ',erf(2/np.sqrt(2)),erf(3/np.sqrt(2)))
