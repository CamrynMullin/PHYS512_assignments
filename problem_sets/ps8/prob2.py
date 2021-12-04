#ps 8 (7) camryn mullin 26026298
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#a)
# p = V - V_ave
# V ~ ln(r)
def get_bc(n):
    #charge at 0,0 where center of grid
    bc = np.zeros([n,n])
    mask = np.zeros([n,n], dtype='bool')
    
    mask[0,:]=True
    mask[-1,:]=True
    mask[:,0]=True
    mask[:,-1]=True
    
    bc[0,:] = 1
    bc[-1,:] = 1
    bc[:,0] = 1
    bc[:,-1] = 1
    return bc, mask

def greens(n):
    dx = np.arange(-n//2,n//2)
    pot = np.zeros([n,n])
    x,y = np.meshgrid(dx,dx)
    r = np.sqrt(x**2 + y**2)
    r[n//2,n//2] = 1 #initial condition
    pot = -np.log(r) #potential from point charge
    pot[n//2,n//2] = 1 #get 1
    return x,y,pot

n = 128
x, y, V = greens(n)
print('The potential at V[1,0] and V[2,0] is', V[n//2,n//2+1], V[n//2,n//2+2])
print('The potential at V[5,0] is', V[n//2,n//2+5])
plt.figure() #fig1
plt.pcolormesh(x,y,V)
plt.colorbar()
plt.savefig('2a_V')
plt.show()

#BONUS  
def average_neighbours(V):
    pot = 0*V
    V = V + np.roll(V,1,axis=0)
    V = V + np.roll(V,-1,axis=0)
    V = V + np.roll(V,1,axis=1)
    V = V + np.roll(V,-1,axis=1)
    return pot*0.25

bc, mask = get_bc(n)
V_B = bc.copy()
for i in range(10*n):
    V_B = average_neighbours(V)  
    V_B[mask] = bc[mask] 
print('The potential at V[1,0] and V[2,0] is', V[n//2,n//2+1], V[n//2,n//2+2])
print('The potential at V[5,0] is', V[n//2,n//2+5])
plt.figure() #fig2
plt.imshow(V_B)
plt.colorbar()
plt.savefig('2a_V_bonus')
plt.show()
#BONUS END

#b)
def rho_to_pot(rho,kernelft):
    tmp = rho.copy()
    tmp = np.pad(tmp,(0,tmp.shape[0]))

    tmpft = np.fft.rfftn(tmp)
    tmp = np.fft.irfftn(tmpft*kernelft)

    tmp = tmp[:rho.shape[0],:rho.shape[1]]
    return tmp


def rho_to_pot_masked(rho,mask,kernelft,return_mat=False):
    rhomat = np.zeros(mask.shape)
    rhomat[mask] = rho
    potmat = rho_to_pot(rhomat,kernelft)
    if return_mat:
        return potmat
    else:
        return potmat[mask]    


def conj_grad(rhs,x0,mask,kernel_ft,fun=rho_to_pot_masked,niter=n):
    Ax = fun(x0,mask,kernel_ft)
    r = rhs-Ax
    p = r.copy()
    x = x0.copy()
    rsqr = np.sum(r*r)
    while rsqr > 1e-16:
        Ap = fun(p,mask,kernel_ft)
        alpha = np.sum(r*r)/np.sum(Ap*p)
        x = x + alpha*p
        r = r - alpha*Ap
        rsqr_new = np.sum(r*r)
        beta = rsqr_new/rsqr
        p = r + beta*p
        rsqr = rsqr_new
    return x

x2, y2, kernel = greens(2*n)
kernel_ft = np.fft.rfft2(kernel)
bc, mask = get_bc(n)

rhs = bc[mask]
x0 = 0*rhs

rho_out = conj_grad(rhs,x0,mask,kernel_ft)

plt.figure()
plt.plot(rho_out[:n]) #fig3
plt.savefig('2b_rho')
plt.show()

#c)
V = rho_to_pot_masked(rho_out,mask,kernel_ft,True)
plt.figure() #fig4
plt.pcolormesh(x,y,V)
plt.colorbar()
plt.savefig('2c_V_box')

def pad_array(array):
    s = np.shape(array)
    new_array = np.zeros((s[0]*2,s[1]*2), dtype=bool)
    new_array[s[0]//2-1:3*s[0]//2-1,s[1]//2-1:3*s[1]//2-1] = array
    return new_array

mask_pad = pad_array(mask)

x4, y4, kernel_pad = greens(4*n)
kernel_ft_pad = np.fft.rfft2(kernel_pad)

V = rho_to_pot_masked(rho_out, mask_pad, kernel_ft_pad,True)
plt.figure() #fig5
plt.pcolormesh(x2,y2,V)
plt.colorbar()
plt.savefig('2c_V_all_space')
 
plt.figure() #fig6
plt.pcolormesh(x2,y2,mask_pad)
plt.savefig('box boundaries')

#it can be seen that the charge is greatest within the box boundaries
#and drops like -ln(r) as you move away from the box
