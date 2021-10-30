#prob 4 pset 5 260926298
import numpy as np
import matplotlib.pyplot as plt
import prob1

def gauss(x,b,sig=False):
    if sig:
        return np.exp(-0.5*(x-b)**2/sig**2)
    return np.exp(-0.5*(x-b)**2)

def conv_safe(f,g):
    f = np.append(f,np.zeros(1))
    g = np.append(g,np.zeros(1))
    while len(f) != len(g):
        if len(f) > len(g):
            print('extending g')
            g = np.append(g,np.zeros(1))
        else:
            print('extending f')
            f = np.append(f,np.zeros(1))    
    h = np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g))
    while np.abs((h[-1] - h[0])) > h.max()/10000:
        print('adding more zeros')
        h = conv_safe(f,g)
    return h
    

x1 = np.linspace(-10,10,1000)
x2 = np.linspace(-10,10,1001)
f = gauss(x1,np.median(x1))
g = np.zeros(len(x2))
g[int(len(f)/2)] = 1

h = conv_safe(f,g)

print('the length of the output is', len(h), 'while the length of f was', len(f),'and the length of g', len(g)) 

plt.figure()
plt.ion()
plt.plot(f,label='f, input')
plt.plot(h, label='h, output')
plt.legend()
plt.savefig('q4.png')
plt.show()
