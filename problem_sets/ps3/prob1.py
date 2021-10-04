#problem 1 Camryn Mullin 260926298
import numpy as np

def fun(x,y):
    #dy/dx = y/(1+x**2)
    global counter
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
c = 1/np.exp(np.arctan(-20)) #1=c*exp(arctan(-20) from intial conditions
y_true = c*np.exp(np.arctan(x))#1/(1+20**2)*(1+x**2)#np.exp(np.arctan(x))
y = np.zeros(n_steps)
y[0] = 1 #y(-20) = 1
for i in range(n_steps-1):
    counter = 0
    h = x[i+1]-x[i]
    y[i+1] = y[i] + rk4_step(fun,x[i],y[i],h) 

print('The error is', np.std(y-y_true), 'with', counter,'function evaluations per step')   

def rk4_stepd(fun,x,y,h):
    y1 = rk4_step(fun,x,y,h)
    y2a = rk4_step(fun,x,y,h/2)
    y2b = rk4_step(fun,x+h/2,y+y2a,h/2)
    y2 = y2a + y2b
    return (4*y2-y1)/3

y = np.zeros(n_steps)
y[0] = 1
for i in range(n_steps-1):
    counter = 0
    h = x[i+1]-x[i]
    y[i+1] = y[i]+rk4_stepd(fun,x[i],y[i],h)

print('The error is', np.std(y-y_true), 'with', counter,'function evaluations per step')  

#print('To evaluate with the same number of function calls as the first rk4_step:')
#y1 = np.zeros(n_steps)
#y1[0] = y_true[0]#1
#y2 = np.zeros(n_steps)
#y2[0] = y_true[0]#1
#counter = 0
#for i in range(n_steps-1):
    #h = x[i+1]-x[i]
    #y1[i+1] = y1[i]+rk4_step(fun,x[i],y1[i],h)
    #y2[i+1] = y2[i]+rk4_stepd(fun,x[i],y2[i],h)
    #k1 = h*fun(x,y)
    #k2 = h*fun(x+h/2,y+k1/2)
    #k3 = h*fun(x+h/2,y+k2/2)
    #k4 = h*fun(x+h,y+k3)
    #y1 = (k1 + 2*k2 + 2*k3 + k4)/6  
    #h = h/2
    #k12a = k1
    #k22a = h*fun(x+h/2,y+k12a/2)
    #k32a = h*fun(x+h/2,y+k22a/2)
    #k42a = h*fun(x+h,y+k32a)
    #y2a = (k12a + 2*k22a + 2*k32a + k42a)/6 
    #x = x + h
    #y = y + y2a
    #k12b = h*fun(x,y)
    #k22b = h*fun(x+h/2,y+k12b/2)
    #k32b = h*fun(x+h/2,y+k22b/2)
    #k42b = h*fun(x+h,y+k32b)
    #y2b = (k12b + 2*k22b + 2*k32b + k42b)/6 
    #y2 = y2a + y2b
    #return (4*y2-y1)/3    