#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt

def model_log2(x,tol,poly='cheb'):
    y = np.log2(x)
    x_rescale = np.interp(x, (x.min(), x.max()), (-1, +1))
    ncoeff = 1
    if poly == 'cheb':
        coeff = np.polynomial.chebyshev.chebfit(x_rescale,y,ncoeff)
        p = np.polynomial.chebyshev.chebval(x_rescale,coeff)
        error = np.abs(p-y)
        while True in (error > tol):
            ncoeff += 1
            coeff = np.polynomial.chebyshev.chebfit(x_rescale,y,ncoeff)
            p = np.polynomial.chebyshev.chebval(x_rescale,coeff)
            error = np.abs(p-y)
    #bonus question 
    elif poly == 'leg':
        coeff = np.polynomial.legendre.legfit(x,y,ncoeff)
        p = np.polynomial.legendre.legval(x,coeff)
        error = np.abs(p-y)
        while True in (error > tol):
            ncoeff += 1
            coeff = np.polynomial.legendre.legfit(x_rescale,y,ncoeff)
            p = np.polynomial.legendre.legval(x_rescale,coeff)
            error = np.abs(p-y)
    return p, error, coeff

def mylog2(x,coeff,poly='cheb'):
    mant_x, expo_x = np.frexp(x) #x = mant_x * 2**expo_x
    mant_e, expo_e = np.frexp(np.exp(1))
    mant_x = np.interp(mant_x, (mant_x.min(), mant_x.max()), (-1, +1))
    mant_e = np.interp(mant_e, (mant_e.min(), mant_e.max()), (-1, +1))
    if poly == 'cheb':
        log2_x = np.polynomial.chebyshev.chebval(mant_x,coeff) + expo_x
        log2_e = np.polynomial.chebyshev.chebval(mant_e,coeff) + expo_e
    elif poly == 'leg':
        log2_x = np.polynomial.legendre.legval(mant_x,coeff) + expo_x
        log2_e = np.polynomial.legendre.legval(mant_e,coeff) + expo_e
    ln = log2_x/log2_e #change of base log rule: ln(x) = log2(x)/log2(e)
    error = ln - np.log(x)
    rms = np.sqrt(np.sum((ln - np.mean(ln))**2)/len(ln))
    return ln, error, rms

npt = 20
x = np.linspace(0.5,1,npt)
tol = 10**(-6)
log2_cheb,err,coeff_cheb = model_log2(x,tol)
print('log2 values for chebyshev fit with a max error of', err.max())
plt.figure()
plt.plot(x, log2_cheb, label='log2 model')
plt.plot(x,np.log2(x),label='log2')
plt.title('Chebyshev for log2')
plt.legend()
plt.show()

ln_cheb,err,rms = mylog2(x,coeff_cheb)
print('ln values for chebyshev fit with a max error of', err.max(), 'and an RMS of', rms)
plt.figure()
plt.plot(x, ln_cheb, label='ln model')
plt.plot(x,np.log(x),label='ln')
plt.title('Chebyshev for ln')
plt.legend()
plt.show()


#bonus
log2_leg,err,coeff_leg = model_log2(x,tol,poly='leg')
print('log2 values for legendre fit with a max error of', err.max())
plt.figure()
plt.plot(x, log2_leg, label='log2 model')
plt.plot(x,np.log2(x),label='log2')
plt.title('Legendre for log2')
plt.legend()
plt.show()

ln2_leg,err,rms = mylog2(x,coeff_leg, poly='leg')
print('ln values for legendre fit with a max error of', err.max(), 'and an RMS of', rms )
plt.figure()
plt.plot(x, ln2_leg, label='ln model')
plt.plot(x,np.log(x),label='ln')
plt.title('Legendre for ln')
plt.legend()
plt.show()
