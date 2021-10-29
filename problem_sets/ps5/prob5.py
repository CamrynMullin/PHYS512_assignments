#prob 5 pset 5 260926298
import numpy as np
import matplotlib.pyplot as plt
#c)
N = 1000
x = np.arange(N-1)
k = 2*np.pi
y = np.sin(2*np.pi*k*x/N)
yft_real = np.fft.rfft(y)
myft = np.zeros(int(N/2),dtype='complex')
for k_i in range(int(N/2)):
    myft[k_i] = (1-np.exp(-2*np.pi*1j*(k_i-k))/(1-np.exp(-2*np.pi*1j*(k_i-k)/N)))/(2j)

plt.figure()
plt.plot(np.abs(yft_real), label='numpy fft')
plt.plot(np.abs(myft), label='analytical')
plt.legend()
plt.savefig('Analytic_estimate.png')

#agreement with delta function:
k = N/2/np.pi
y = np.sin(2*np.pi*k*x/N)
yft_real = np.fft.rfft(y)
myft = np.zeros(int(N/2),dtype='complex')
for k_i in range(int(N/2)):
    myft[k_i] = (1-np.exp(-2*np.pi*1j*(k_i-k))/(1-np.exp(-2*np.pi*1j*(k_i-k)/N)))/(2j)

plt.figure()
plt.plot(np.abs(yft_real), label='numpy fft')
plt.plot(np.abs(myft), label='analytical')
plt.legend()
plt.savefig('delta_pure_sine_wave.png')

#d)
window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
yft_wind = np.fft.rfft(y*window)
plt.figure()
plt.plot(np.abs(yft_real), label='numpy fft')
plt.plot(np.abs(myft), label='analytical, no window')
plt.plot(np.abs(yft_wind), label='ft with window')
plt.legend()
plt.savefig('Window_function.png')

