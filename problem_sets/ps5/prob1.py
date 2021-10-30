#260926298 prob1 pset 5
import numpy as np
import matplotlib.pyplot as plt

def gauss(x,b,sig=False):
    if sig:
        return np.exp(-0.5*(x-b)**2/sig**2)
    return np.exp(-0.5*(x-b)**2)

def conv(arr, n):
    f = arr
    #let g be a delta function centered around n
    g = np.zeros(len(f))
    g[int(n)] = 1
    return np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g),len(f))


def main():
    x = np.linspace(-10,10,1000)
    f = gauss(x,np.median(x))
    h = conv(f,int(len(f)/2))
    
    plt.figure()
    plt.ion()
    plt.plot(x,f,label='origional gauss')
    plt.plot(x,h, label='shifted gauss')
    plt.legend()
    plt.savefig('q1.png')
    plt.show()

if __name__ == "__main__":
    main()