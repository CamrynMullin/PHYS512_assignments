#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt
#import corner
import prob1

def get_chisq(pars,y,noise,fun):
    model = fun(pars)
    return np.sum(((y-model)/noise)**2)

def mcmc(pars,step_size,x,y,fun,noise,nstep=5000):
    current_chisq = get_chisq(pars,y,noise,fun)
    npar = len(pars)
    chain = np.zeros([nstep,npar])
    chain_errs = np.zeros([nstep,npar])
    chisq_vec = np.zeros(nstep)
    for i in range(nstep):
        print('step', i)
        trial_pars = pars + step_size*np.random.randn(npar)
        trial_chisq = get_chisq(trial_pars,y,noise,fun)
        delta_chisq = trial_chisq - current_chisq
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1) < accept_prob #always accept if chisq decrease
        if accept:
            pars = trial_pars
            current_chisq = trial_chisq
        chain[i,:] = pars
        chain_errs[i,:] = np.std(chain, axis=0)
        chisq_vec[i] = current_chisq
        print('pars are', pars, 'with errors', chain_errs[i,:])
        print('chisq is', current_chisq)
    return chain,chain_errs,chisq_vec

#def main():
model_fun = prob1.get_spectrum  
x = prob1.ell
y = prob1.spec
errs = prob1.errs
#matrix = np.loadtxt("curvture_matrix.txt")
step_size = np.loadtxt('planck_fit_params.txt')[:,1]
pars = np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
chain, errs, chisq = mcmc(pars,step_size,x,y,model_fun,errs)
print('The final best fit values were', chain[-1], 'with errors', errs[-1], 'and chisq', chisq[-1])
    
data = np.column_stack([chisq,chain])
np.savetxt("planck_chain.txt", data)
    
#computing dark energy
h = chain[-1][0]/100
dark_energy = 1 - chain[-1][1]/h**2 - chain[-1][2]/h**2
dark_err = errs[-1][1]/h**2 + errs[-1][2]/h**2
print('Mean value of dark energy is', dark_energy, 'with error', dark_err)
    
#plotting
plt.figure()
plt.ion()
plt.plot(chisq)
plt.title('Chi square from MCMC')
plt.savefig('Chisq_prob3.png')
plt.show()

plt.ion()
planck_binned = np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned = 0.5*(planck_binned[:,2]+planck_binned[:,3]);
plt.figure()
plt.plot(x,y, label='data')
plt.plot(x,model_fun(pars), label='Old Fit'); plt.plot(x,model_fun(chain[-1]), label='MCMC fit')
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='error bars')
plt.legend()
plt.savefig('q3_power_spectrum.png')
plt.show()

plt.ion()
plt.figure()
plt.plot(x,y-model_fun(pars),'.', label='Old Fit'); plt.plot(x,y-model_fun(chain[-1]), '.',label='MCMC fit')
plt.errorbar(x,np.zeros(len(x)),prob1.errs,fmt='.', label='error bar')
plt.legend()
plt.savefig('q3_residuals.png')
plt.show()


chain_pars = np.array(chain)
#plt.figure()
#plt.ion()
labels = ['H0', 'Ω_b h^2', 'Ω_c h^2', 'τ', 'A_s', 'n_s']

#corner.corner(chain_pars[20:],labels=labels, show_titles=True,title_fmt='.2f')
#plt.savefig('corner_prob3.png')
#plt.show()
    
#forier transform of each parameter
plt.figure()
for i in range(len(chain_pars[0])):
    plt.loglog(np.abs(np.fft.rfft(chain_pars[:,i])), label='{}'.format(labels[i]))
plt.title('Paramater fourier tranform')
plt.legend()
plt.savefig('fourier_prob3.png')
plt.show()
    
#if __name__ == "__main__":
    #main()
