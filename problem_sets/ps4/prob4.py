#problem 4 260926298
import numpy as np
import matplotlib.pyplot as plt
import prob1

def get_chisq(y,model,noise):
    return np.sum(((y-model)/noise)**2)

def ndiff(fun,x1,x2,dx):
    deriv = (fun(x1)-fun(x2))/(2*dx)
    return deriv 

def update_lamda(lamda,success):
    if success:
        lamda /= 1.5
        if lamda < 0.5 :
            lamda = 0
    else:
        if lamda == 0:
            lamda = 1
        else:
            lamda = lamda*1.5**2
    return lamda

def get_matricies(m,fun,x,y,noise,N_inv,prior,prior_err):
    model = fun(m)
    derivs = np.zeros([len(x),len(m)])
    for i in range(len(m)):
        x1 = m.copy()
        x2 = m.copy()
        dx = 0.01*m[i]
        x1[i] = m[i] + dx 
        x2[i] = m[i] - dx
        derivs[:,i] += ndiff(fun,x1,x2,dx)
    res = y - model
    lhs = derivs.T@N_inv@derivs
    rhs = derivs.T@(N_inv@res)
    chisq = res.T@N_inv@res + get_chisq(m,prior,prior_err)
    return chisq,lhs,rhs

def linv(mat, lamda):
    return np.linalg.inv(mat + lamda*np.diag(np.diag(mat)))

def fit_lm(m0,fun,x,y,noise,prior,prior_err):
    m = m0.copy()
    N = np.eye(len(noise))*noise**2
    N_inv = np.linalg.inv(N)
    lamda = 0
    chisq,lhs,rhs = get_matricies(m,fun,x,y,noise,N_inv,prior,prior_err)
    delta_chisq = chisq
    while delta_chisq > 0.01:
        dm = linv(lhs,lamda)@rhs
        chisq_new,lhs_new,rhs_new = get_matricies(m+dm,fun,x,y,noise,N_inv,prior,prior_err)
        if chisq_new < chisq:
            print('chisq decreased!')
            lhs = lhs_new
            rhs = rhs_new
            m += dm            
            delta_chisq = chisq - chisq_new
            if lamda == 0 and np.abs(delta_chisq) <= 0.01:
                print('found covariance matrix', matrix)
                matrix = np.linalg.inv(lhs)
                return matrix
            chisq = chisq_new
            lamda = update_lamda(lamda,True)
        else:
            lamda = update_lamda(lamda,False)
        print('chisq is ',chisq,' with pars ',m, 'and step', dm)
    matrix = np.linalg.inv(lhs)
    print('found covariance matrix', matrix)
    return matrix 

def mcmc(pars,step_size,x,y,fun,noise,prior,prior_err,nstep=5000):
    current_chisq = get_chisq(y,fun(pars),noise) + get_chisq(pars,prior,prior_err)
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
        trial_chisq = get_chisq(y,fun(trial_pars),noise) + get_chisq(trial_pars,prior,prior_err)
        delta_chisq = trial_chisq - current_chisq
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1) < accept_prob #always accept if chisq decrease, sometimes accept otherwise
        if accept:
            pars = trial_pars
            current_chisq = trial_chisq
        chain[i,:] = pars
        chain_errs[i,:] = np.std(chain, axis=0)
        chisq_vec[i] = current_chisq
        print('pars are', pars, 'with error', chain_errs[i,:])
        print('chisq is', current_chisq)
    return chain,chain_errs,chisq_vec

def importance_sample(matrix,prior,prior_err):
    nsamp = matrix.shape[0]
    npar = matrix.shape[1]
    chisq_vec = np.zeros(nsamp)
    for i in range(nsamp): 
        #chisq value for each row of params
        chisq_vec[i] = get_chisq(matrix[i,:],prior,prior_err)
    chisq_vec -= np.mean(chisq_vec)
    weight = np.exp(-0.05*chisq_vec)          
    par_mean = np.empty(npar)
    for i in range(npar):
        #average value of weighted paramater
        par_mean[i] = np.sum(weight*matrix[:,i])/np.sum(weight)    
    return par_mean   

model_fun = prob1.get_spectrum  
x = prob1.ell
y = prob1.spec
noise = prob1.errs
tau = 0.0540
tau_err = 0.0074
#pars = np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
pars = np.array([69,0.022,0.12,tau,2.1e-9,0.95])
prior = np.zeros(len(pars))
prior[3] = tau
prior_err = np.zeros(len(pars)) + 1e20
prior_err[3] = tau_err

#step size via importance sampling covariance matrix
matrix = fit_lm(pars,model_fun,x,y,noise,prior,prior_err)
step_size = np.linalg.cholesky(matrix)

#run MCMC
chain, errs, chisq = mcmc(pars,step_size,x,y,model_fun,noise,prior,prior_err)

pars_final = np.mean(chain[-1000:],axis=0)
errs_final = np.mean(errs[-1000:],axis=0)
chisq_final = np.mean(chisq[-1000:],axis=0)
print('The final best fit values were', pars_final, 'with errors', errs_final, 'and chisq', chisq_final)
data = np.column_stack([chisq,chain])
np.savetxt("planck_chain_tauprior.txt", data)

#importance sample from chain in prob3
chain_old = np.loadtxt("planck_chain.txt")
chain_old = chain_old.T[1:].T #remove first column which is chisq values
pars_sample = importance_sample(chain_old,prior,prior_err)
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
plt.plot(x,model_fun(pars), label='Old Fit'); plt.plot(x,model_fun(pars_final), label='MCMC fit')
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='error bars')
plt.legend()
plt.savefig('q4_power_spectrum.png')
plt.show()

chain_pars = np.array(chain)
labels = ['H_0', 'Ω_b h^2', 'Ω_c h^2', 'τ', 'A_s', 'n_s']
    
#forier transform of each parameter
plt.figure()
for i in range(len(chain_pars[0])):
    plt.loglog(np.abs(np.fft.rfft(chain_pars[:,i])), label='${}$'.format(labels[i]))
plt.title('Paramater fourier tranform')
plt.legend()
plt.savefig('fourier_prob4.png')
plt.show()