#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h
import corner
import prob1
import prob2 

def get_chisq(pars,x,y,noise,fun):
    model = fun(pars)
    return np.sum (((y-model)/noise)**2)

def mcmc(pars,step_size,x,y,model,noise,nstep=1000):
    chi_cur = get_chisq(pars,x,y,noise,model)
    npar = len(pars)
    chain = np.zeros([nstep,npar])
    chivec = np.zeros(nstep)
    for i in range(nstep):
        trial_pars = pars + step_size*np.random.randn(npar)
        trial_chisq = get_chisq(trial_pars,x,y,noise,model)
        delta_chisq = trial_chisq - chi_cur
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1) < accept_prob
        if accept:
            pars = trial_pars
            chi_cur = trial_chisq
        chain[i,:] = pars
        chivec[i] = chi_cur
    return chain,chivec

def main():
    model_fun = prob1.get_spectrum  
    x = prob1.ell
    y = prob1.spec
    errs = prob1.errs
    step_size = prob2.matrix    
    pars = np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
    chain, chisq = mcmc(pars,step_size,x,y,model_fun,errs)
    #step_size_new = np.std(chain,axis=0)
    #starting_pars = np.mean(chain,axis=0)+3*np.random.randn(npar)*step_size_new
    #chain2, chisq2 = mcmc(starting_pars,step_size_new,x,y,model_fun,errs)
    
    data = np.column_stack([chain,chisq])
    np.savetxt("planck_chain.txt", data, fmt=['%d','%d'])
    dark_energy = 1 - chain[-1][1]/h**2 - chain[-1][2]/h**2
    print('Mean value of dark energy is', dark_energy, 'with error', chisq[-1])
    
    #plotting
    plt.ion()
    labels = ['H0', 'Ω_b h^2', 'Ω_c h^2', 'τ', 'A_s', 'n_s']
    corner.corner(pars,labels=labels, show_titles=True,title_fmt='.2f')
    plt.show()
    
    #forier transform of each parameter
    for i in range(len(pars[0])):
        plt.loglog(np.abs(np.fft.rfft(pars[:,i])))
    plt.show()
    
if __name__ == "__main__":
    main()
