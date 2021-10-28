#260926298 pset 5 problem 6
import numpy as np
import matplotlib.pyplot as plt
#6 b)
N = 1000
y = np.cumsum(np.random.randn(N))
yft = np.fft.rfft(y/y.sum()) 
k = np.arange(N/2)

scale = np.array([1/k_i**2 for k_i in k if k_i!=0])
plt.figure()
plt.ion()
plt.plot(np.abs(yft), label='power spectrum')
plt.plot(scale, label='$1/k^2$')
#plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('rw_ps.png')
