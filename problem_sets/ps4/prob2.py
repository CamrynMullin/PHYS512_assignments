#problem 2 2609262298
import numpy as np
import prob1

def ndiff(fun,x1,x2,dx): #version of differentiator from ps1 problem 2 
    deriv = (fun(x1)-fun(x2))/(2*dx)
    return deriv  

def chisq_fun(y,model,noise):
    return np.sum(((y-model)/noise)**2) 

#newtons
def fit_newton(m0,fun,x,y,noise):
    m = m0.copy()
    N = np.eye(len(noise))*noise**2
    N_inv = np.linalg.inv(N)
    chisq_old = 0 #chisq_fun(y,fun(m),noise)
    delta_chisq = 1 #random initialization #chisq_old
    while delta_chisq > 0.01:
        derivs = np.zeros([len(x),len(m)])
        for i in range(len(m)):
            x1 = m.copy()
            x2 = m.copy()
            dx = 0.01*m[i]
            x1[i] = m[i] + dx 
            x2[i] = m[i] - dx
            derivs[:,i] += ndiff(fun,x1,x2,dx)
        model = fun(m)
        res = y - model
        chisq = chisq_fun(y,model,noise)
        lhs = derivs.T@N_inv@derivs
        rhs = derivs.T@N_inv@res
        dm = np.linalg.inv(lhs)@rhs
        m += dm
        print('chisq is ',chisq,' with step ',dm) 
        delta_chisq = chisq - chisq_old
        chisq_old = chisq
    matrix = np.linalg.inv(lhs)   
    errs = np.sqrt(np.diag(matrix))
    return m, errs, chisq, matrix
    
fun = prob1.get_spectrum
x = prob1.ell
y = prob1.spec
errs = prob1.errs

m0 = np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]) #initial guess
m_fit, m_errs, chisq, matrix = fit_newton(m0,fun,x,y,errs)

print('the best fit params are', m_fit, 'with errors', m_errs)
data = np.column_stack([m_fit,m_errs])
np.savetxt("planck_fit_params.txt", data)
np.savetxt("curvture_matrix.txt", matrix)
    
#bonus
#m0[2] = 0 #set dark matter density set to zero
#m_fit, matrix, chisq = fit_newton(m0,fun,x,y,errs)
#m_errs = np.mean(matrix, axis = 0)
#print('the best fit with dark matter density set to zero is', m_fit)
#print('the corresponding chisq value is', chisq)
#data = np.column_stack([m_fit,m_errs])
##print('data', data)
#np.savetxt("planck_fit_params_nodm.txt", data, fmt=['%d','%d'])    
