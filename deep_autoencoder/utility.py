# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# Configuration of the SAEs
def load_config():      
    
    aux1=np.loadtxt('cnf_sae.csv')
    aux2=np.loadtxt('cnf_softmax.csv')
    class ParamSae:
        pinv=int(aux1[0])
        actFunc=int(aux1[1])
        maxIter=int(aux1[2])
        batchSize=int(aux1[3])
        lr=aux1[4]
        nEncoders=aux1[5:]

    class ParamSft:
        maxIter=int(aux2[0])
        lr=aux2[1]
        batchSize=int(aux2[2])

    params_sae=ParamSae()
    params_sft=ParamSft()
    return(params_sae,params_sft)

# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
# STEP 1: Feed-forward of AE
def dae_forward(x,...):
    ...
    return()    


#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))
# STEP 2: Feed-Backward for DAE
def gradW(a,w2):   
    ...
    ...    
    return(...)        

# Update DAE's weight via mAdam
def updW_madam():
        ...    
    return(...v)
# Update Softmax's weight via mAdam
def updW_sft_rmsprop(w,v,gw,mu):
    ...    
    return(w,v)

# Softmax's gradient
def gradW_softmax(x,y,a):        
    ya   = y*np.log(a)
    ...    
    return(gW,Cost)

# Calculate Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))


# save weights DL and costo of Softmax
def save_w_dl(...):    
    ...
    
