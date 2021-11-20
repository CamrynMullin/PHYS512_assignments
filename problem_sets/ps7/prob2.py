#problem 2 ps7 Camryn Mullin 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

#write a rejection method to generate exponential derivatives from another distribution
#lorenzien, gaussian or power for bounding distribution?
#show histogram of deviates matches up with exponential curve
#how efficient can you make this generator?

def make_hist(x,bins):
    hist, bin_edges = np.histogram(x,bins) #x values, edges
    centers = 0.5*(bin_edges[1:] + bin_edges[:-1]) #wanna compare at center
    hist = hist/hist.sum()
    return hist, centers #normalized histogram and centers

n = 100000
q = np.random.rand(n)
# exponential: x = ae^(-at) = CDF^-1(q), CDF: e^(-t) = q --> t = -ln(q)
#in this assignment set a = 1 for simplisity
t = -np.log(q)

#histogram
bins = np.linspace(1,20,101)
hist_exp, centers_exp = make_hist(t,bins)
pred_exp = np.exp(-centers_exp) #PDF
pred_exp = pred_exp/pred_exp.sum()

plt.figure()
plt.plot(centers_exp, hist_exp, '*')
plt.plot(centers_exp, pred_exp, 'r')
plt.title('Exponential Deviates')
plt.ylabel('exp')
plt.savefig('q2_prediced_exponential.png')
plt.show()

#REJECTION
def reject(y):
    h = np.random.rand(n)*1.22 #height
    #for exponential, prob = exp(-x) goes to exp(-tan(y))/cos^2(y)  
    accept = h < np.exp(-np.tan(y))/np.cos(y)**2
    x_use = np.tan(y[accept])    
    return x_use, accept

#try lorentzian:
def lorentzian_x(n):
    q = np.pi*(np.random.rand(n) - 0.5) #range from -pi/2 - +pi/2
    return np.tan(q) #x

def lorentzian_y(x):
    return 1/(1+x**2)

x = lorentzian_x(n)
#y = 1/(1+x^2)
y_lor = 1/(1 + x**2)*np.random.rand(n)*3 #constants for scaling

hist_lor, centers_lor = make_hist(x,bins)
my_exp = np.exp(-centers_lor)
my_lor = lorentzian_y(centers_lor)*3 # x3 to ensure it's higher than exp

plt.figure()
plt.plot(x, y_lor,'.',color='pink', label = 'lorentzian hist')
plt.plot(centers_lor, my_lor,'b', label = 'lorentzian')
plt.plot(centers_lor, my_exp,'r', label = 'exponential')
plt.xlim(0,20)
plt.ylim(0,1.5) #zoom in to show that lorenzian always above exponential
plt.title('Lorentzien vs Exponential Deviates')
plt.legend()
plt.savefig('lorentzian_vs_exp_curve.png')
plt.show()

#rejection     
x_use, accept_lor = reject(y_lor)

hist_lor, centers_lor = make_hist(x_use,bins)
pred_exp = np.exp(-centers_lor) #PDF
pred_exp = pred_exp/pred_exp.sum()

plt.figure()
plt.ion()
plt.plot(centers_lor, hist_lor, '*', label = 'histogram of deviates')
plt.plot(centers_lor, pred_exp, 'r', label = 'predicted exponential')
plt.legend()
plt.title('Exponential Deviates from Lorentzien Rejection')
plt.savefig('lorentzian_deviates_hist.png')
plt.show()


#trying gaussian:
def gauss_x(n):
    q = np.pi*(np.random.rand(n) - 0.5) 
    return erfinv(q*np.sqrt(2/np.pi))*np.sqrt(2) #see math in PDF

def gauss_y(x):
    return np.exp(-0.5*x**2)

x = gauss_x(n)
y_gauss = gauss_y(x)*np.random.rand(n)*10

hist_gauss, centers_gauss = make_hist(x,bins)
my_exp = np.exp(-centers_gauss)
my_gauss = gauss_y(centers_gauss)*10 #gaussian will not go above for all x!!

plt.figure()
plt.plot(x,y_gauss,'.',color = 'pink', label = 'power gauss')
plt.plot(centers_gauss, my_gauss,'b', label = 'gauss')
plt.plot(centers_gauss, my_exp,'r', label = 'exponential')
plt.xlim(0,20)
plt.title('Gaussian vs Exponential')
plt.legend()
plt.savefig('gaussian_vs_exp_curve')
plt.show()

#rejection     
x_use, accept_gauss = reject(y_gauss)

hist_gauss, centers_gauss = make_hist(x_use,bins)
pred_exp = np.exp(-centers_gauss) #PDF
pred_exp = pred_exp/pred_exp.sum()

plt.figure()
plt.ion()
plt.plot(centers_gauss, hist_gauss, '*', label = 'histogram of deviates')
plt.plot(centers_gauss, pred_exp, 'r', label = 'predicted exponential')
plt.title('Exponential Deviates from Gaussin Rejection')
plt.legend()
plt.savefig('gauss_deviates_hist.png')
plt.show()

#trying power law:
alpha = 1.5
#PDF t^-alpha, q = 1- T^(1-alpha) --> T = (1-q)^(1/(1-alpha))
t = (1 - q)**(1/(1 - alpha))
y_power = t**(-alpha)*np.random.rand(n)*3

hist_power, centers_power = make_hist(t,bins)
my_power = centers_power**(-alpha)*3 #x3 for scalling
my_exp = np.exp(-centers_power)

plt.figure()
plt.plot(t,y_power,'.', color = 'pink', label = 'power hist')
plt.plot(centers_power, my_power,'b', label = 'power')
plt.plot(centers_power, my_exp,'r', label = 'exponential')
plt.xlim(0,20)
plt.title('Power-law vs Exponential Deviates')
plt.legend()
plt.savefig('power_vs_exp_curve.png')
plt.show()

#y_power = np.pi*(np.random.rand(n)-0.5)
t_use, accept_power = reject(y_power)

hist_power, centers_power = make_hist(t_use,bins)
pred_exp = np.exp(-centers_power) #PDF
pred_exp = pred_exp/pred_exp.sum()

plt.figure()
plt.ion()
plt.plot(centers_power, hist_power, '*', label = 'histogram of deviates')
plt.plot(centers_power, pred_exp, 'r', label = 'predicted exponential')
plt.title('Exponential Deviates from Power-Law Rejection')
plt.legend()
plt.savefig('power_deviates_hist.png')
plt.show()

#EFFCIENCY
def eff(y, accept):
    used = y[accept]
    return len(used)/len(y) * 100

lor_eff = eff(y_lor, accept_lor)
gauss_eff = eff(y_gauss, accept_gauss)
power_eff = eff(y_power, accept_power)
efficiency = [lor_eff, gauss_eff, power_eff]
names = ['Lorentzien', 'Gaussian', 'Power Law']
print(efficiency)
best = [i for i,val in enumerate(efficiency) if val == max(efficiency)]
print('the most efficient method is', names[best[0]], 'with effieciency', efficiency[best[0]], '%')
