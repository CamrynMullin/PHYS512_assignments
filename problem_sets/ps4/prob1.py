#prob 1 CamrynMullin 260926298
import numpy as np
import matplotlib.pyplot as plt
import camb

def get_spectrum(pars,lmax=3000):
    #print('params are', pars)
    H0 = pars[0]
    ombh2 = pars[1]
    omch2 = pars[2]
    tau = pars[3]
    As = pars[4]
    ns = pars[5]
    pars_camb = camb.CAMBparams()
    pars_camb.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars_camb.InitPower.set_params(As=As,ns=ns,r=0)
    pars_camb.set_for_lmax(lmax,lens_potential_accuracy=0)
    results = camb.get_results(pars_camb)
    powers = results.get_cmb_power_spectra(pars_camb,CMB_unit='muK')
    cmb = powers['total']
    tt = cmb[:,0]
    return tt[2:len(spec)+2] #remove monopoles and return power specturm

planck = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell = planck[:,0] #multipole
spec = planck[:,1] #variance in multipole
errs = 0.5*(planck[:,2]+planck[:,3]); #1 sigma error

#above can be use in multiple problems
def main(): #parts that are question 1 specific
    #from test script
    pars1 = np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])  
    model1 = get_spectrum(pars1)
    resid1 = spec-model1
    chisq1 = np.sum((resid1/errs)**2)
    n1 = len(resid1)-len(pars1)
    print("chisq from test script is",chisq1,"for",n1," degrees of freedom.")

    #new paramaters
    pars2 = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])  
    model2 = get_spectrum(pars2)
    resid2 = spec-model2
    chisq2 = np.sum((resid2/errs)**2)
    n2 = len(resid2)-len(pars2)
    print("chisq with new parametrs is",chisq2,"for",n2," degrees of freedom.")
    
    #plotting
    plt.ion()
    planck_binned = np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
    errs_binned = 0.5*(planck_binned[:,2]+planck_binned[:,3]);
    plt.figure()
    plt.plot(ell,model1, label='test pars'); plt.plot(ell,model2, label='new pars')
    plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='error bars')
    plt.legend()
    plt.savefig('q1_power_spectrum.png')
    plt.show()
    
if __name__ == "__main__":
    main()

