#problem 1 Camryn Mullin 260926298
import numpy as np

def fun(x,y):
    global counter #count num function evaluations
    counter += 1       
    return y/(1+x**2)

def rk4_step(fun,x,y,h):
    k1 = h*fun(x,y)
    k2 = h*fun(x+h/2,y+k1/2)
    k3 = h*fun(x+h/2,y+k2/2)
    k4 = h*fun(x+h,y+k3)
    return (k1 + 2*k2 + 2*k3 + k4)/6 

n_steps = 200
x = np.linspace(-20,20,n_steps)  
c = 1/np.exp(np.arctan(-20)) #1=c*exp(arctan(-20)) from intial conditions
y_true = c*np.exp(np.arctan(x))
y = np.zeros(n_steps)
y[0] = 1 #y(-20) = 1
for i in range(n_steps-1):
    counter = 0
    h = x[i+1]-x[i]
    y[i+1] = y[i] + rk4_step(fun,x[i],y[i],h) 

print('The error is', np.std(y-y_true), 'with', counter,'function evaluations per step')   

#def rk4_stepd(fun,x,y,h):
    #y1 = rk4_step(fun,x,y,h)
    #y2a = rk4_step(fun,x,y,h/2)
    #y2b = rk4_step(fun,x+h/2,y+y2a,h/2)
    #y2 = y2a + y2b
    #return (4*y2-y1)/3

def rk4_stepd(fun,x,y,h):
    #y1 = rk4_step(fun,x,y,h)
    k1 = h*fun(x,y)
    k2 = h*fun(x+h/2,y+k1/2)
    k3 = h*fun(x+h/2,y+k2/2)
    k4 = h*fun(x+h,y+k3)
    y1 = (k1 + 2*k2 + 2*k3 + k4)/6     
    #y2a = rk4_step(fun,x,y,h/2)
    k1a = k1/2
    k2a = h/2*fun(x+h/4,y+k1a/2)
    k3a = h/2*fun(x+h/4,y+k2a/2)
    k4a = h/2*fun(x+h/2,y+k3a)
    y2a = (k1a + 2*k2a + 2*k3a + k4a)/6  
    #y2b = rk4_step(fun,x+h/2,y+y2a,h/2)
    x += h/2
    y += y2a
    k1b = h/2*fun(x,y)
    k2b = h/2*fun(x+h/4,y+k1b/2)
    k3b = h/2*fun(x+h/4,y+k2b/2)
    k4b = h/2*fun(x+h/2,y+k3b)
    y2b = (k1b + 2*k2b + 2*k3b + k4b)/6  
    y2 = y2a + y2b
    return (4*y2-y1)/3    
    
y = np.zeros(n_steps)
y[0] = 1
for i in range(n_steps-1):
    counter = 0
    h = x[i+1]-x[i]
    y[i+1] = y[i]+rk4_stepd(fun,x[i],y[i],h)

print('The error is', np.std(y-y_true), 'with', counter,'function evaluations per step')  

print('To evaluate with the same number of function calls as the first rk4_step:')
print('The allowed number of function evaluations is', (n_steps-1)*4)
n_steps = (n_steps-1)*4 // 11
x = np.linspace(-20,20,n_steps) 
y_true = c*np.exp(np.arctan(x))
y = np.zeros(n_steps)
y[0] = 1
counter = 0
for i in range(n_steps-1):
    h = x[i+1]-x[i]
    y[i+1] = y[i]+rk4_stepd(fun,x[i],y[i],h)
print('The error for rk2_stepd is', np.std(y-y_true), 'with', counter, 'evaluations') 