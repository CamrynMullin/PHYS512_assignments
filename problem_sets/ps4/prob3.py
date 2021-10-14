#problem 3 260926298
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h
import corner
import prob1
#import prob2

def get_chisq(pars,x,y,noise,fun):
    model = fun(pars)
    return np.sum(((y-model)/noise)**2)

def mcmc(pars,step_size,x,y,fun,noise,nstep=1000):
    current_chisq = get_chisq(pars,x,y,noise,fun)
    npar = len(pars)
    chain = np.zeros([nstep,npar])
    chisq_vec = np.zeros(nstep)
    for i in range(nstep):
        print('step', i)
        trial_pars = pars + step_size*np.random.randn(npar)
        trial_chisq = get_chisq(trial_pars,x,y,noise,fun)
        delta_chisq = trial_chisq - current_chisq
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1) < accept_prob
        if accept:
            pars = trial_pars
            current_chisq = trial_chisq
        chain[i,:] = pars
        chisq_vec[i] = current_chisq
    return chain,chisq_vec

def main():
    model_fun = prob1.get_spectrum  
    x = prob1.ell
    y = prob1.spec
    errs = prob1.errs
    #step_size = np.loadtxt("curvture_matrix.txt")
    step_size = np.loadtxt('planck_fit_params.txt')[:,1]
    #print(step_size)
    pars = np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
    chain, chisq = mcmc(pars,step_size,x,y,model_fun,errs)
    #step_size_new = np.std(chain,axis=0)
    #starting_pars = np.mean(chain,axis=0)+3*np.random.randn(npar)*step_size_new
    #chain2, chisq2 = mcmc(starting_pars,step_size_new,x,y,model_fun,errs)
    
    data = np.column_stack([chain,chisq])
    np.savetxt("planck_chain.txt", data)
    h = chain[-1][0]/100
    dark_energy = 1 - chain[-1][1]/h**2 - chain[-1][2]/h**2
    print('Mean value of dark energy is', dark_energy, 'with error', chisq[-1])
    
    ##plotting
    #plt.figure()
    #plt.ion()
    #labels = ['H0', 'Ω_b h^2', 'Ω_c h^2', 'τ', 'A_s', 'n_s']
    #corner.corner(pars,labels=labels, show_titles=True,title_fmt='.2f')
    #plt.show()
    
    ##forier transform of each parameter
    #plt.figure()
    #for i in range(len(pars[0])):
        #plt.loglog(np.abs(np.fft.rfft(pars[:,i])))
    #plt.show()
    
if __name__ == "__main__":
    main()
