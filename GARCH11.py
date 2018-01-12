
# coding: utf-8

# # Estimating a GARCH(1,1) from A to Z
# 
# $Gilles \space HACHEME$

# $\textbf{Garch(1,1)}  :  \epsilon_{t} = v_{t} \sqrt{h_{t}}\\
# h_{t} = a_{0} + a_{1} \epsilon_{t-1}^2 + b_{1}h_{t-1} \\
# With \space v_{t} \sim  \mathcal{N}(0, h_{t})
# 	$

# Iterating $h_{t}$ until order t, we get : 
# 
# $h_{t} = a_{0} \frac{1-b_{1}^t}{1 - b_{1}} + a_{1} \sum_{i=0}^{t-1} b_{1}^i \epsilon_{t-1-i}^2 + b_{1}^t h_{0}$
# 
# We didn't use here any stationarity condition. And for this tutorial, we will suppose $h_{0} = 0$ 
# 
# For memory we would use in the stationary assumption case : 
# 
# $h_{t} =  \frac{a_{0}}{1 - b_{1}} + a_{1} \sum_{i=0}^{t-1} b_{1}^i \epsilon_{t-1-i}^2$
# 
# For t sufficiently large

# In[27]:

import numpy as np

#Simulating some values for epsilon square
Epsquare_t1=np.array([2,4,6,9,0,4,7])
T=len(Epsquare_t1)  #Number of observations


# $
# \textbf{Computing the Likelihood function} : \\\\
#     L(\theta) = \prod_{t=1}^{T} \frac{1}{2\pi h_{t}} \exp{\frac{-\epsilon_{t}^2}{2h_{t}}} \\ 
#     With \space \theta = (a_{0}, a_{1}, b_{1}) \\\\
# \textbf{So the Log-Likelihood is} : \\\\
#     logL(\theta) = -\frac{1}{2} \sum_{t=1}^{T} [log2\pi + log h_{t} + \frac{\epsilon_{t}^2}{h_{t}}]$
# 

# # The Log-Likelihood function

# In[42]:

def Logl(a_0,a_1,b_1,T):
    L=[]
    for t in range(2,T+1):
        
        Bi=[(b_1)**i for i in range(0,t)]    
        
        
        Epsquare_t1i = Epsquare_t1[:len(Bi)].reshape(t,1)
        
        Bi=np.array(Bi).reshape(1,t)
        
        h_t= a_0*(1 - b_1**t)/(1-b_1) + a_1*np.dot(Bi,Epsquare_t1i)
        
        logl= np.log(2*np.pi) + np.log(h_t) + Epsquare_t1[t-1]/(h_t)
        
        L.append(logl)
    

    
    
    return -(1/2)*np.sum(L)


# In[43]:

a_0,a_1,b_1=0.1,0.7,0.2
Logl(a_0,a_1,b_1,T)


# # First and Second derivative

# In[46]:

def first_dev(f,x):
    h1=0.0001
    fx=f(x)
    fxh1=f(x+h1)
    
    return (fxh1-fx)/h1
    


# In[47]:

def second_dev(f,x):
    h2=0.0001
    return (first_dev(f,x+h2) - first_dev(f,x))/h2


# In[48]:

def func(x):
    return x**3


# In[49]:

first_dev(func,1)


# In[50]:

second_dev(func,2)


# # Finding the optimal parameters

# We are going to use the New Raphson algorithm to get successive local maximums :
# 
# - Give initial guess to $\theta_{0} = (a_{0}^0, a_{1}^0,b_{1}^0)$
#     
# - Getting a local maximum : 
#    for instance for $a_{0}$, we can fix the values of $a_{1}$ and $b_{1}$ and compute its optimal value : $a_{0}^{new} = a_{0}^{old} - \frac{\partial logL(a_{0})}{\partial a_{0}}|_{a_{0} = a_{0}^{old}} [\frac{\partial^2 logL(a_{0})}{\partial a_{0}^2}|_{a_{0} = a_{0}^{old}}]^{-1}$
#    
#    After each iteration, we check the sign of $diff = logL(a_{0}^{new})-logL(a_{0}^{old})$. 
#    * If diff > 0, then we update the value of $a_{0}$ to get a new value. NB : The last $a_{0}^{new}$ becomes the new $a_{0}^{old}$.
#    * If diff $\leq$ 0, then we stop the iteration and retain the current $a_{0}^{old} = a_{0}^*$.
# - After getting the optimal value $a_{0}^*$, we use it to get the optimal value $a_{1}^*$ by the same process. And finally we use theses two optimal values to get the optimal value $b_{1}^*$

# In[44]:

#Lets' choose the initial values

a0_0,a0_1,b0_1=0.1,0.7,0.2
#Starting Log-likelihood
Logl(a0_0,a0_1,b0_1,T)


# In[62]:

#Let's first optimize a_0

#Here we fix a_1 = a0_1 and b_1 = b0_1 
def Logl_a_0(a_0):
    return Logl(a_0,a0_1,b0_1,T)

a_0=a0_0
nb_iter=0
while True:
    L0=Logl_a_0(a_0)

    a1_0 = a_0 - first_dev(Logl_a_0,a_0)/second_dev(Logl_a_0,a_0)   #New value of a_0

    L1=Logl_a_0(a1_0)   #New value of the log-likelihood
    nb_iter+=1
    
    print("Nb iter : {}".format(nb_iter),"a_0 = {}".format(round(a_0,3)),"L1-L0 = {}".format(round(L1-L0,3)),sep="; ")
    if (L1-L0)<0:    #If the sign of the difference if negative
        break        #Stop the iterating process
    
    a_0 = a1_0   # Otherwise, update the value of a_0

a_0_opt = a_0     #Optimal value


# In[63]:

print("initial value : {}".format(a0_0),"Optimized value : {}".format(round(a_0_opt,3)),sep="\n")


# In[64]:

#Let's  optimize a_1

#Here we fix a_0 = a_0_opt and b_1 = b0_1 
def Logl_a_1(a_1):
    return Logl(a_0_opt,a_1,b0_1,T)

a_1=a0_1
nb_iter=0
while nb_iter<10:
    L0=Logl_a_1(a_1)

    a1_1 = a_1 - first_dev(Logl_a_1,a_1)/second_dev(Logl_a_1,a_1)

    L1=Logl_a_1(a1_1)   #New value of the log-likelihood
    nb_iter+=1
    
    
    if (L1-L0)<0:
        break
    
    print("Nb iter : {}".format(nb_iter),"a_1 = {}".format(round(a_1,3)),"L1-L0 = {}".format(round(L1-L0,3)),sep="; ")
    a_1 = a1_1
    
a_1_opt = a_1


# In[65]:

print("initial value : {}".format(a0_1),"Optimized value : {}".format(round(a_1_opt,3)),sep="\n")


# In[66]:

#Let's first optimize b_1

#Here we fix a_0 = a_0_opt and a_1 = a_1_opt  
def Logl_b_1(b_1):
    return Logl(a_0_opt,a_1_opt,b_1,T)

b_1=b0_1
nb_iter=0
while nb_iter<10:
    L0=Logl_b_1(b_1)

    b1_1 = b_1 - first_dev(Logl_b_1,b_1)/second_dev(Logl_b_1,b_1)

    L1=Logl_b_1(b1_1)   #New value of the log-likelihood
    nb_iter+=1
    
    
    if (L1-L0)<0:
        break
    
    print("Nb iter : {}".format(nb_iter),"b_1 = {}".format(round(b_1,3)),"L1-L0 = {}".format(round(L1-L0,3)),sep="; ")
    
    b_1 = b1_1
    
b_1_opt = b_1 


# In[67]:

print("initial value : {}".format(b0_1),"Optimized value : {}".format(round(b_1_opt,3)),sep="\n")

