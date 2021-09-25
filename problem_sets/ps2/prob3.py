#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt

def model_log2(x,tol,poly='chebyshev'):
    y = np.log2(x)
    ncoeff = 1
    
    if poly == 'chebyshev':
        x_rescale = np.interp(x, (x.min(), x.max()), (-1, +1))
        coeff = np.polynomial.chebyshev.chebfit(x_rescale,y,ncoeff)
        p = np.polynomial.chebyshev.chebval(x_rescale,coeff)
        error = np.abs(p-y)
        while True in (error > tol):
            ncoeff += 1
            coeff = np.polynomial.chebyshev.chebfit(x_rescale,y,ncoeff)
            p = np.polynomial.chebyshev.chebval(x_rescale,coeff)
            error = np.abs(p-y)

    #bonus question 
    elif poly == 'legendre':
        coeff = np.polynomial.legendre.legfit(x,y,ncoeff)
        p = np.polynomial.legendre.legval(x,coeff)
        error = np.abs(p-y)
        while True in (error > tol):
            ncoeff += 1
            coeff = np.polynomial.legendre.legfit(x,y,ncoeff)
            p = np.polynomial.legendre.legval(x,coeff)
            error = np.abs(p-y)
    
    print('Needed', ncoeff, 'terms to evalualte {}'.format(poly))
    rms = np.sqrt(np.sum((p - np.mean(p))**2)/len(p))
    return p, error, rms, coeff

def mylog2(x,coeff,poly='chebyshev'):
    mant, expo = np.frexp(x)
    
    if poly == 'chebyshev':
        x_rescale = np.interp(mant, (mant.min(), mant.max()), (-1, +1))
        log2_x = np.polynomial.chebyshev.chebval(x_rescale,coeff) + expo
        
    #Bonus
    elif poly == 'legendre':
        log2_x = np.polynomial.legendre.legval(mant,coeff) + expo
    
    ln = log2_x/np.log2(np.exp(1)) #change of base log rule: ln(x) = log2(x)/log2(e)
    error = np.abs(ln - np.log(x)) #np.log is the numpy ln
    rms = np.sqrt(np.sum((ln - np.mean(ln))**2)/len(ln))
    return ln, error, rms

npt = 1000
x = np.linspace(0.5,1,npt)
tol = 10**(-6)
log2_cheb,err,rms_cheb, coeff_cheb = model_log2(x,tol)
print('log2 values for chebyshev fit with a max error of', err.max(), 'and an RMS of', rms_cheb)
plt.figure()
plt.plot(x, log2_cheb, label='log2 model')
plt.plot(x,np.log2(x),'--', label='np.log2')
plt.title('Chebyshev for log2')
plt.legend()
plt.show()

x = np.linspace(0.01,100,npt)
ln_cheb,err,rms = mylog2(x,coeff_cheb)
print('ln values for chebyshev fit with a max error of', err.max(), 'and an RMS of', rms)
plt.figure()
plt.plot(x, ln_cheb, label='ln model')
plt.plot(x,np.log(x),'--', label='np.log')
plt.title('Chebyshev for ln')
plt.legend()
plt.show()

#bonus
x = np.linspace(0.5,1,npt)
log2_leg,err,rms_leg, coeff_leg = model_log2(x,tol,poly='legendre')
print('log2 values for legendre fit with a max error of', err.max(),'and an RMS of', rms_leg)
plt.figure()
plt.plot(x, log2_leg, label='log2 model')
plt.plot(x,np.log2(x),'--',label='np.log2')
plt.title('Legendre for log2')
plt.legend()
plt.show()

x = np.linspace(0.01,100,npt)
ln2_leg,err,rms = mylog2(x,coeff_leg, poly='legendre')
print('ln values for legendre fit with a max error of', err.max(), 'and an RMS of', rms )
plt.figure()
plt.plot(x, ln2_leg, label='ln model')
plt.plot(x,np.log(x),'--',label='np.log')
plt.title('Legendre for ln')
plt.legend()
plt.show()
