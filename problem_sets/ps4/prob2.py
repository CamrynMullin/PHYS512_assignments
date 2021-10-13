#problem 2 2609262298
import numpy as np
import prob1
    
#def ndiff(fun,x,ind): #version of differentiator from ps1 problem 2
    #pars1 = x.copy()
    #pars2 = x.copy()
    #dx = 0.01*x[ind]
    #pars1[ind] = x[ind]+dx
    #pars2[ind] = x[ind]-dx
    #f1, f2 = fun(pars1), fun(pars2)
    #deriv = (f1-f2)/(2*dx)
    #return deriv
def ndiff(fun,x,x1,x2,dx,double = False): #version of differentiator from ps1 problem 2
    f1, f2 = fun(x1), fun(x2)
    if double:
        deriv = (f1 + f2 - 2*fun(x))/(dx**2)
    else:
        deriv = (f1-f2)/(2*dx)
    return deriv  

def chisq_fun(y,model,noise):
    return np.sum(((y-model)/noise)**2) 

#newtons
def fit_newton(m0,fun,x,y,noise):
    m = m0.copy()
    N = np.eye(len(noise))*noise
    N_inv = np.linalg.inv(N)
    chisq_old = 0 #chisq_fun(y,fun(m),noise)
    delta_chisq = 1 #random initialization #chisq_old
    while delta_chisq > 0.01:
        derivs = np.zeros([len(x),len(m)])
        for i in range(len(m)):
            x1 = m.copy()
            x2 = m.copy()
            dx = 0.01*m[i]
            x1[i],x2[i] = m[i] + dx, m[i] - dx
            derivs[:,i] += ndiff(fun,m,x1,x2,dx)
            #derivs[:,i] += ndiff(fun,m,i)
        model = fun(m)
        res = y - model
        chisq = chisq_fun(y,model,noise)
        lhs = derivs.T@N_inv@derivs
        rhs = derivs.T@N_inv@res
        dm = np.linalg.pinv(lhs)@rhs
        m += dm
        print('chisq is ',chisq,' with step ',dm) 
        delta_chisq = chisq - chisq_old
        chisq_old = chisq
    #eps = 2**-52 
    #dx = eps**(1/3)    
    #curvature_matrix = ndiff(chisq_fun,model,model+dx,model-dx,dx,double=True)
    return m, dm, chisq, lhs#, curvature_matrix
    
fun = prob1.get_spectrum
x = prob1.ell
y = prob1.spec
errs = prob1.errs

m0 = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) #initial guess
m_fit, m_errs, chisq, matrix = fit_newton(m0,fun,x,y,errs)

#def main():
#m_errs = np.mean(matrix, axis = 0)
print('the best fit params are', m_fit, 'with errors', m_errs)
data = np.column_stack([m_fit,m_errs])
#print('data', data)
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
    
#if __name__ == "__main__":
    #main()

#lm
#def update_lamda(lamda, success):
    #if success:
        #lamda /= 1.5
        #if lamda < 0.5:
            #lamda = 0
    #else:
        #if lamda==0:
            #lamda = 1
        #else:
            #lamda *= 1.5**2
    #return lamda  

#def get_matrices(m,fun,x,y,Ninv=None):
    #model = fun(m,x)
    #derivs = ndiff(fun,m,x) 
    #res = y - model
    #if Ninv is None:
            #lhs = derivs.T@derivs
            #rhs = derivs.T@res
            #chisq = np.sum(res**2)
    #else:
        #lhs = derivs.T@Ninv@derivs
        #rhs = derivs.T@(Ninv@res)
        #chisq = res.T@Ninv@res
    #return chisq,lhs,rhs

#def inv(A, lamda):
    #mat = A + lamda*np.diag(np.diag(A))
    #return np.linalg.lin(mat)
    
#def fit_lm(m,fun,x,y,Ninv=None,niter=10,chitol=0.01):
    #lamda = 0
    #chisq,lhs,rhs = get_matrices(m,fun,x,y,Ninv)
    #for i in range(niter):
        #lhs_inv = linv(lhs,lamda)
        #dm = lhs_inv@rhs
        #chisq_new,lhs_new,rhs_new = get_matrices(m+dm,fun,x,y,Ninv)
        #if chisq_new < chisq:                                                     
            #if lamda==0 and (np.abs(chisq-chisq_new)<chitol):
                #return m+dm
            #chisq = chisq_new
            #lhs = lhs_new
            #rhs = rhs_new
            #m += dm
            #lamda = update_lamda(lamda,True)
        #else:
            #lamda = update_lamda(lamda,False)
        #print('on iteration ',i,' chisq is ',chisq,' with step ',dm,' and lamda',lamda)
    #return m, lhs