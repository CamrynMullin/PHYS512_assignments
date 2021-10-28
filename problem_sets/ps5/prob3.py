#prob 3 pset 5 260926298
import numpy as np
import matplotlib.pyplot as plt
import prob1
import prob2

def cor_shift(arr1,arr2,n):
    #shift array
    arr1 = prob1.conv(arr1,n)
    arr2 = prob1.conv(arr2,n)
    return prob2.corr(arr1,arr2) #correlation
 

x = np.linspace(-10,10,1000)
f = prob1.gauss(x,np.median(x))
#normalization
f = f/f.sum()
corr_f1 = cor_shift(f,f,int(len(x)/2))
corr_f2 = cor_shift(f,f,int(len(x)/100))

plt.figure()
plt.ion()
plt.plot(x,f,label='gaussian')
plt.plot(x,corr_f1, label='shifted by len(x)/2')
plt.plot(x,corr_f2, label='shifted by len(x)/100')
plt.legend(loc='upper right')
plt.savefig('gauss_corr_shifted.png')
plt.show()

#correlation does not seem to depend on shift
