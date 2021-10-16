#problem 4 260926298
import numpy as np
import matplotlib.pyplot as plt
import prob1

def get_chisq(y,model,noise=None):
    if noise is None:
        return np.sum(((y-model))**2)
    return np.sum(((y-model)/noise)**2)

#def prior_chisq(pars, prior, prior_err):
    #return np.sum(((pars - prior)/prior_err)**2)

def importance_sample(matrix,prior,prior_err=None,mean=False):
    if mean: #if we want the mean on each param in the matrix
        nsamp = matrix.shape[0]
        chisq_vec = np.zeros(nsamp)
        for i in range(nsamp): 
            #chisq value for each row of params
            chisq_vec[i] = get_chisq(matrix[i,:],prior,prior_err)
        chisq_vec -= np.mean(chisq_vec)
        weight = np.exp(-0.05*chisq_vec)          
        par_mean = np.empty(nsamp)
        for i in range(nsamp):
            #average value of weighted paramater
            par_mean[i] = np.sum(weight*matrix[:,i])/np.sum(weight)    
        return par_mean  
    ##otherwise return the importance sampled matrix
    #chisq_mat = np.zeros([len(matrix),matrix.shape[0]])
    #for i in range(len(matrix)):
        #for j in range(matrix.shape[0]):
            #chisq_mat[i,j] = get_chisq(matrix[i,j],prior[i,j])
    #chisq_mat -= np.mean(chisq_mat)
    #weight = np.exp(0.05*chisq_mat)      
    #sampled_matrix =  weight*matrix 
    #return sampled_matrix   

def mcmc(pars,step_size,x,y,fun,noise,prior,prior_err,step=5000):
    current_chisq = get_chisq(y,pars,noise) + get_chisq(pars,prior,prior_err)
    npar = len(pars)
    chain = np.zeros([nstep,npar])
    chain_errs = np.zeros([nstep,npar])
    chisq_vec = np.zeros(nstep)
    for i in range(nstep):
        print('step', i)
        trial_pars = pars + step_size@np.random.randn(npar)
        trial_chisq = get_chisq(y,fun(trial_pars),noise) + get_chisq(pars,prior,prior_err)
        delta_chisq = trial_chisq - current_chisq
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1) < accept_prob #always accept if chisq decrease, sometimes accept otherwise
        if accept:
            pars = trial_pars
            current_chisq = trial_chisq
        chain[i,:] = pars
        chain_errs[i,:] = np.std(chain, axis=0)
        chain_errs[i,3] = tau_err
        chisq_vec[i] = current_chisq
        print('pars are', pars)
        print('chisq is', current_chisq)
    return chain,chain_errs,chisq_vec

model_fun = prob1.get_spectrum  
x = prob1.ell
y = prob1.spec
errs = prob1.errs
tau = 0.0540
tau_err = 0.0074
pars = np.array([69,0.022,0.12,tau,2.1e-9,0.95])

prior = np.zeros(len(pars))
prior[3] = tau
prior_err = np.zeros(len(pars)) + 1e20
prior_err[3] = tau_err

#step size via importance sampling covariance matrix
matrix = np.loadtxt("curvture_matrix.txt")
#cov_prior = np.linalg.cholesky(matrix)
#step_size = importance_sample(matrix,cov_prior) 
step_size = np.linalg.cholesky(matrix)
chain, errs, chisq = mcmc(pars,step_size,x,y,model_fun,errs,prior,prior_err)
print('The final best fit values were', chain[-1], 'with errors', errs[-1], 'and chisq', chisq[-1])
data = np.column_stack([chisq,chain])
np.savetxt("planck_chain_tauprior.txt", data)

#importance sample from chain in prob3
chain_old = np.loadtxt("planck_chain.txt")
chain_old = chain_old.T[1:].T #remove first column which is chisq values
pars_sample = importance_sample(chain_old,prior,prior_err,mean=True)
sample_errs = np.std(chain_old, axis=0)
sample_chisq = get_chisq(y,model_fun(pars_sample),noise)
print('Pars from sampling chain in prob3 are', pars_sample,'with errors',sample_errs, 'and chisq', sample_chisq)

#plotting
plt.figure()
plt.ion()
plt.plot(chisq)
plt.title('Chi square from MCMC')
plt.savefig('Chisq_prob4.png')
plt.show()

plt.ion()
planck_binned = np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned = 0.5*(planck_binned[:,2]+planck_binned[:,3]);
plt.figure()
plt.plot(x,y, label='data')
plt.plot(x,model_fun(pars), label='Old Fit'); plt.plot(x,model_fun(chain[-1]), label='MCMC fit')
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='error bars')
plt.legend()
plt.savefig('q4_power_spectrum.png')
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
plt.savefig('fourier_prob4.png')
plt.show()