#prob 3 pset 5 260926298
import numpy as np
import matplotlib.pyplot as plt
import prob1
import prob2

def cor_shift(arr1,arr2,n):
    #shift array
    arr1 = prob1.conv(arr1,n)
    return prob2.corr(arr1,arr2) #correlation

x = np.linspace(-10,10,1000)
f = prob1.gauss(x,np.median(x))
corr_unshift = prob2.corr(f,f)
corr_f1 = cor_shift(f,f,int(len(x)/2))
corr_f2 = cor_shift(f,f,int(len(x)/4))
corr_f3 = cor_shift(f,f,int(len(x)/8))
corr_f4 = cor_shift(f,f,int(len(x)-1))

plt.figure()
plt.ion()
plt.plot(x,corr_unshift, label='unshifted')
plt.plot(x,corr_f1, label='shifted by len(x)/2')
plt.plot(x,corr_f2, label='shifted by len(x)/4')
plt.plot(x,corr_f3, label='shifted by len(x)/8')
plt.plot(x,corr_f4, label='shifted by len(x)')
plt.legend(loc='upper right')
plt.savefig('q3.png')
plt.show()

