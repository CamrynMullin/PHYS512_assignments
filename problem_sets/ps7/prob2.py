#problem 2 ps7 Camryn Mullin 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
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
# exponential: x = e^(-t) = CDF^-1(q), CDF: e^(-t) = q --> t = -ln(q)
t = -np.log(q)

#histogram
bins = np.linspace(1,20,101)
hist_exp, centers_exp = make_hist(t,bins)
pred_exp = np.exp(-centers_exp) #PDF
pred_exp = pred_exp/pred_exp.sum()

#plt.figure()
#plt.plot(centers_exp, hist_exp, '*')
#plt.plot(centers_exp, pred_exp, 'r')
#plt.ylabel('exp')
##plt.savefig('exponential.png')
#plt.show()

#try lorentzian rejection:

def lorentzian_x(n):
    q = np.pi*(np.random.rand(n) - 0.5) #range from -pi/2 - +pi/2
    return np.tan(q) #x

def lorentzian_y(x):
    return 1/(1+x**2)

x = lorentzian_x(n)
y_lor = 1.5/(1 + x**2)*np.random.rand(n)*2

hist_lor, centers_lor = make_hist(x,bins)
pred_lor = lorentzian_y(centers_lor)
pred_lor = pred_lor/pred_lor.sum()

#plt.figure()
#plt.plot(centers_lor, hist_lor, '*')
#plt.plot(centers_lor, pred_lor, 'r')
#plt.ylabel('lorenrzien')
##plt.savefig('lorentzien.png')
#plt.show()

my_exp = np.exp(-centers_lor)
my_lor = lorentzian_y(centers_lor)*3

plt.figure()
plt.plot(x,y_lor,'.')
plt.plot(centers_lor, my_lor,'b', label = 'lorenzian')
plt.plot(centers_lor, my_exp,'r', label = 'exponential')
plt.legend()
plt.show()

##accepting values
#accept = y_lor < np.exp(-x) #if y falls under exponential
#x_use = x[accept]

#hist_lor, centers_lor = make_hist(x_use,bins)
#pred_lor = lorentzian_y(centers_lor)
#pred_lor = pred_lor/pred_lor.sum()

#plt.figure()
#plt.ion()
#plt.plot(centers_lor, hist_lor, '*')
#plt.plot(centers_lor, pred_lor, 'r')
##plt.savefig('lorentzien.png')
#plt.show()

#rejection     
y = np.pi*(np.random.rand(n)-0.5)
h = np.random.rand(n)*1.22 #height, must be larger than curve
#for exponential, prob=exp(-x) goes to exp(-tan(y))/cos^2(y)  
accept = h < np.exp(-np.tan(y))/np.cos(y)**2
x_use = np.tan(y[accept])

hist_lor, centers_lor = make_hist(x_use,bins)
pred_exp = np.exp(-centers_lor) #PDF
pred_exp = pred_exp/pred_exp.sum()

plt.figure()
plt.ion()
plt.plot(centers_lor, hist_lor, '*')
plt.plot(centers_lor, pred_exp, 'r')
#plt.savefig('lorentzien.png')
plt.show()


##trying gaussian:
#def gauss_x(n):
    #q = np.pi*(np.random.rand(n) - 0.5) 
    #return erfinv(q*np.sqrt(2/np.pi))*np.sqrt(2)

#def gauss_y(x):
    #return np.exp(-0.5*x**2)

#x = gauss_x(n)
#y_gauss = gauss_y(x)

#hist_gauss, centers_gauss = make_hist(x,bins)
#pred_gauss = gauss_y(centers_gauss)
#pred_gauss = pred_gauss/pred_gauss.sum()

##plt.figure()
##plt.plot(centers_gauss, hist_gauss, '*')
##plt.plot(centers_gauss, pred_gauss, 'r')
##plt.ylabel('gauss')
###plt.savefig('lorentzien.png')
##plt.show()

#my_exp = np.exp(-centers_lor)
#my_gauss = gauss_y(centers_gauss)*3

#plt.figure()
#plt.plot(x,y_gauss,'.')
#plt.plot(centers_gauss, my_gauss,'b', label = 'gauss')
#plt.plot(centers_gauss, my_exp,'r', label = 'exponential')
#plt.legend()
#plt.show()

##rejection     
#y = np.pi*(np.random.rand(n)-0.5)
#h = np.random.rand(n)*5 #height, must be larger than curve
##for exponential, prob=exp(-x) goes to exp(-tan(y))/cos^2(y)  
#accept = h < np.exp(-np.tan(y))/np.cos(y)**2
#x_use = np.tan(y[accept])

#hist_gauss, centers_gauss = make_hist(x_use,bins)
#pred_exp = np.exp(-centers_gauss) #PDF
#pred_exp = pred_exp/pred_exp.sum()
#plt.figure()
#plt.ion()
#plt.plot(centers_gauss, hist_gauss, '*')
#plt.plot(centers_gauss, pred_exp, 'r', label = 'gauss')
##plt.savefig('lorentzien.png')
#plt.show()


##trying power law:

#alpha = 1.5
##PDF t^-alpha, q = 1- T^(1-alpha) --> T = (1-q)^(1/(1-alpha))
#t = (1 - q)**(1/(1 - alpha))
#y_power = t**(-alpha)*np.random.rand(n)

#hist_power, centers_power = make_hist(t,bins)
#pred_power = centers_power**(-alpha)
#pred_power = pred_power/pred_power.sum()

##plt.figure()
##plt.plot(centers_power, hist_power, '*')
##plt.plot(centers_power, pred_power, 'r')
##plt.show()

#my_power = centers_power**(-alpha)*3

#plt.figure()
#plt.plot(t,y_power,'.')
#plt.plot(centers_lor, my_lor,'b', label = 'power')
#plt.plot(centers_lor, my_exp,'r', label = 'exponential')
#plt.legend()
#plt.show()

#accept = y_power < np.exp(-t)
#t_use = t[accept]

#hist_power, centers_power = make_hist(t_use,bins)
#pred_power = centers_power**(-alpha)
#pred_power = pred_power/pred_power.sum()

#plt.figure()
#plt.ion()
#plt.plot(centers_power, hist_power, '*')
#plt.plot(centers_power, pred_power, 'r')
##plt.savefig('lorentzien.png')
#plt.show()


