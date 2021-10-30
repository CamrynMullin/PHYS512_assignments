#prob 5 pset 5 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#c)
N = 100
x = np.arange(N)
k = 1.5
y = np.sin(2*np.pi*k*x/N)
yft_real = np.fft.rfft(y)
myft = np.zeros(N,dtype='complex')
for k_i in range(N):
    num1 = 1-np.exp(-2*np.pi*1j*(k_i-k))
    den1 = 1-np.exp(-2*np.pi*1j*(k_i-k)/N)
    num2 = 1-np.exp(-2*np.pi*1j*(k_i+k))
    den2 = 1-np.exp(-2*np.pi*1j*(k_i+k)/N)
    myft[k_i] = (2j)**-1*((num1/den1)-(num2/den2))
myft = myft[:N//2+1]
plt.figure()
plt.plot(np.abs(yft_real), label='numpy fft')
plt.plot(np.abs(myft), '--', label='analytical')
plt.legend()
plt.savefig('q5c.png')

residuals = myft - yft_real
error = np.abs(np.std(residuals))
print('The error between the analytic estimate and the numpy fft', error)

#agreement with delta function:
k = N*0.5*np.pi
y = np.sin(2*np.pi*k*x/N)
yft_true = np.fft.rfft(y)
myft = np.zeros(N,dtype='complex')
for k_i in range(N):
    num1 = 1-np.exp(-2*np.pi*1j*(k_i-k))
    den1 = 1-np.exp(-2*np.pi*1j*(k_i-k)/N)
    num2 = 1-np.exp(-2*np.pi*1j*(k_i+k))
    den2 = 1-np.exp(-2*np.pi*1j*(k_i+k)/N)
    myft[k_i] = (2j)**-1*((num1/den1)-(num2/den2))  
myft = myft[:N//2+1]     
delta = signal.unit_impulse(N//2+1,idx=np.argmax(np.abs(myft)))*np.max(np.abs(myft))

plt.figure()
plt.plot(np.abs(yft_true), label='numpy fft')
plt.plot(np.abs(myft), '--',label='analytical')
plt.plot(np.abs(delta), '--',label='delta funciton')
plt.legend()
plt.savefig('q5c_delta_compare.png')

residuals = yft_true - delta
error = np.abs(np.std(residuals))
print('The error between fft of pure sin wave and a delta function', error)

#d)
window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
yft_wind = np.fft.rfft(y*window)
delta = signal.unit_impulse(N//2+1,idx=np.argmax(np.abs(yft_wind)))*np.max(np.abs(yft_wind))
plt.figure()
plt.plot(np.abs(yft_true), label='numpy fft')
plt.plot(np.abs(myft),'--',label='analytical')
plt.plot(np.abs(yft_wind), '--', label='numpy fft with window')
plt.plot(np.abs(delta), '--',label='delta funciton')
plt.legend()
plt.savefig('q5d.png')

residuals = yft_wind - delta
error = np.abs(np.std(residuals))
print('The error between windowed fft of pure sin wave and a delta function', error)

#e)
window_fft = np.fft.fft(window)
expected_window_fft = np.zeros(len(window_fft))
expected_window_fft[0] = N/2
expected_window_fft[1] = -N/4
expected_window_fft[-1] = -N/4

plt.figure()
plt.plot(np.abs(window_fft), label = 'window fft')
plt.plot(np.abs(expected_window_fft), '--', label='[N/2-N/4 0...0 -N/4]')
plt.legend()
plt.savefig('q5e.png')

residuals = window_fft - expected_window_fft
error = np.abs(np.std(residuals))
print('The error between windowed fft and the expected is', error)


#showing combinations with neihbours
yft_smooth = 1/2*yft_true - 1/4*np.roll(yft_true,1) - 1/4*np.roll(yft_true,-1)
plt.figure()
plt.plot(np.abs(yft_wind), label = 'window')
plt.plot(np.abs(yft_smooth), '--', label='[N/2-N/4 0...0 -N/4]')
plt.legend()
plt.savefig('q5e_neighbours.png')

residuals = yft_smooth - yft_wind
error = np.abs(np.std(residuals))
print('The error between applying a window directly in fft, and applying what it scales as aferwards', error)
