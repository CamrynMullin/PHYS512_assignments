#prob 5 pset 5 260926298
import numpy as np
import matplotlib.pyplot as plt
#c)
N = 1000
x = np.arange(N)
k = 1.5
k_p = 1
y = np.sin(2*np.pi*k*x/N)
yft_real = np.fft.fft(y/y.sum())
yft_real = np.fft.fftshift(yft_real)
myft = np.zeros(N,dtype='complex')
for x_i in range(N):
    myft[x_i] = np.sum(y*np.exp(-2*np.pi*1j*k_p*x_i/N))

plt.figure()
plt.plot(np.abs(yft_real), label='real')
plt.plot(np.abs(myft)/np.abs(myft).sum(), label='analytical')
plt.legend()
plt.savefig('Analytic_estimate.png')

#d)
window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
yft_wind = np.fft.fft(y/y.sum()*window)
plt.figure()
plt.plot(np.abs(yft_real), label='ft without window')
plt.plot(np.abs(yft_wind), label='ft with window')
plt.legend()
plt.savefig('Window_function.png')

