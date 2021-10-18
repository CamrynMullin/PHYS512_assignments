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
        trial_pars = pars + step_size@np.random.randn(npar)
        while trial_pars[3] < 0:
            print('tau negative')
            trial_pars = pars + step_size@np.random.randn(npar)
        trial_chisq = get_chisq(trial_pars,y,noise,fun)
        delta_chisq = trial_chisq - current_chisq
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1) < accept_prob #always accept if chisq decrease, sometimes accept otherwise
        if accept:
            pars = trial_pars
            current_chisq = trial_chisq
        chain[i,:] = pars
        chain_errs[i,:] = np.std(chain, axis=0)
        chisq_vec[i] = current_chisq
        print('pars are', pars, 'with errors', chain_errs[i,:])
        print('chisq is', current_chisq)
    return chain,chain_errs,chisq_vec

model_fun = prob1.get_spectrum  
x = prob1.ell
y = prob1.spec
noise = prob1.errs
matrix = np.loadtxt("curvture_matrix.txt")
step_size = np.linalg.cholesky(matrix)
pars = np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
chain, errs, chisq = mcmc(pars,step_size,x,y,model_fun,noise)

pars_final = np.mean(chain[-1000:],axis=0)
errs_final = np.mean(errs[-1000:],axis=0)
chisq_final = np.mean(chisq[-1000:],axis=0)
print('The final best fit values were', pars_final, 'with errors', errs_final, 'and chisq', chisq_final)
    
data = np.column_stack([chisq,chain])
np.savetxt("planck_chain.txt", data)
    
#computing dark energy
h = pars_final[0]/100
dark_energy = 1 - pars_final[1]/h**2 - pars_final[2]/h**2
dark_err = errs_final[1]/h**2 + errs_final[2]/h**2
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
plt.plot(x,model_fun(pars), label='Old Fit'); plt.plot(x,model_fun(pars_final), label='MCMC fit')
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='error bars')
plt.legend()
plt.savefig('q3_power_spectrum.png')
plt.show()

chain_pars = np.array(chain)
labels = ['H_0', 'Ω_b h^2', 'Ω_c h^2', 'τ', 'A_s', 'n_s']
    
#forier transform of each parameter
plt.figure()
for i in range(len(chain_pars[0])):
    plt.loglog(np.abs(np.fft.rfft(chain_pars[:,i])), label='${}$'.format(labels[i]))
plt.title('Parameter fourier transform')
plt.legend()
plt.savefig('fourier_prob3.png')
plt.show()
