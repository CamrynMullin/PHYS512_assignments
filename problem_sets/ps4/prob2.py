#problem 2 2609262298
import numpy as np
import prob1
    
def ndiff(fun,x,ind): #version of differentiator from ps1 problem 2
    pars1 = x.copy()
    pars2 = x.copy()
    dx = 0.01*x[ind]
    pars1[ind] = x[ind]+dx
    pars2[ind] = x[ind]-dx
    f1, f2 = fun(pars1), fun(pars2)
    deriv = (f1-f2)/(2*dx)
    return deriv
    
#newtons
def fit_newton(m0,fun,x,y,noise, niter = 2):
    m = m0.copy()
    N = np.eye(len(noise))*noise
    N_inv = np.linalg.inv(N)
    #tol_cur = np.zeros([len(m),len(m)])
    #tol_cur += tol+1
    #while tol_cur.any() > tol:
    for j in range(niter):
        derivs = np.zeros([len(x),len(m)])
        for i in range(len(m)):
            derivs[:,i] += ndiff(fun,m,i)
        #print(derivs)
        model = fun(m)
        res = y - model
        chisq = np.sum((res/noise)**2)
        lhs = derivs.T@N_inv@derivs
        rhs = derivs.T@N_inv@res
        dm = np.linalg.pinv(lhs)@rhs
        m += dm
        #print(lhs)
        #tol_cur = lhs
        print('chisq is ',chisq,' with step ',dm) 
    return m, dm, chisq
    
fun = prob1.get_spectrum
x = prob1.ell
y = prob1.spec
errs = prob1.errs

m0 = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) #initial guess
m_fit, m_errs, chisq = fit_newton(m0,fun,x,y,errs)

#def main():
#m_errs = np.mean(matrix, axis = 0)
print(m_fit, m_errs)
data = np.column_stack([m_fit,m_errs])
#print('data', data)
np.savetxt("planck_fit_params.txt", data)
#np.savetxt("curvture_matrix.txt", matrix)
    
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