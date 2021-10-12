#problem 2 2609262298
import numpy as np
import prob1

def ndiff(fun,x): #differentiator from ps1 problem 2
    eps = 2**-52 
    dx = eps**(1/3)
    f1, f2 = fun(x+dx), fun(x-dx)
    deriv = (f1-f2)/(2*dx) #the derivative
    return deriv

#newtons
def fit_newton(m,fun,x,y,niter=10):
    for i in range(niter):
        model = fun(m)
        derivs = ndiff(fun,m,x)
        res = y - model
        lhs = derivs.T@derivs
        rhs = derivs.T@res
        dm = np.linalg.inv(lhs)@rhs
        m += dm
        chisq = np.sum(res**2)
        print('on iteration ',i,' chisq is ',chisq,' with step ',dm)
    return m, res, lhs
    
fun = prob1.get_spectrum
x = prob1.ell
y = prob1.spec
errs = prob1.errs

m0 = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) #initial guess
m_fit, res, matrix = fit_newton(m0,fun,x,y)
def main():
    data = np.column_stack([m_fit,res])
    print('data', data)
    #np.savetxt("planck_fit_params.txt", data, fmt=['%d','%d'])
    
if __name__ == "__main__":
    main()


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