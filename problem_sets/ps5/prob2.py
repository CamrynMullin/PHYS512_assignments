#prob2 pset 5 260926298
import numpy as np
import matplotlib.pyplot as plt

def gauss(x,b,sig=False):
    if sig:
        return np.exp(-0.5*(x-b)**2/sig**2)
    return np.exp(-0.5*(x-b)**2)

def corr(arr1,arr2):
    dft_f = np.fft.rfft(arr1)
    conj_dft_g = np.conj(np.fft.rfft(arr2))
    return np.fft.irfft(dft_f*conj_dft_g,len(arr1))

def main():
    x = np.linspace(-10,10,1000)
    f = gauss(x,np.median(x))  
    #normalization
    corr_f = corr(f,f)
    
    plt.figure()
    plt.ion()
    plt.plot(np.abs(corr_f), label='correlation')
    plt.legend(loc='upper right')
    plt.savefig('q2.png')
    plt.show()

if __name__ == "__main__":
    main()