#problem 4
import numpy as np
import prob1
import prob2
import prob3

model_fun = prob1.get_spectrum  
x = prob1.ell
y = prob1.spec
errs = prob1.errs
step_size = prob2.matrix
tau = 0.0540
tau_err = 0.0074
pars = np.array([69,0.022,0.12,tau,2.1e-9,0.95])
mcmc = prob3.mcmc
#step_size = a new step size
chain, chisq = mcmc(pars,step_size,x,y,model_fun,errs)
data = np.column_stack([chain,chisq])
np.savetxt("planck_chain_tauprior.txt", data, fmt=['%d','%d'])