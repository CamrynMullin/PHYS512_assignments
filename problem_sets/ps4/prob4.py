#problem 4 260926298
import numpy as np
import prob1
import prob3

model_fun = prob1.get_spectrum  
x = prob1.ell
y = prob1.spec
errs = prob1.errs
#step_size = a new step size
tau = 0.0540
tau_err = 0.0074
pars = np.array([69,0.022,0.12,tau,2.1e-9,0.95])
mcmc = prob3.mcmc
chain, chisq = mcmc(pars,step_size,x,y,model_fun,errs)
data = np.column_stack([chain,chisq])
np.savetxt("planck_chain_tauprior.txt", data)

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