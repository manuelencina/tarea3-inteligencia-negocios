# My Utility : auxiliars functions

import numpy  as np

def relu(x):
    return np.maximum(0, x)

def lrelu(x):
    return np.maximum(0.05 * x, x)

def elu(x):
    a = 0.1
    return np.where(x > 0, x, a * (np.exp(x) - 1))

def selu(x):
    a = 1.6732
    d = 1.0507
    return d * np.where(x > 0, x, a * (np.exp(x) - 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation_function(param, x, deriv=None):
    functions = {
        1: lambda x: np.maximum(0, x),
        2: lambda x: np.maximum(0.05 * x, x),
        3: lambda x: np.maximum(0.1 * (np.exp(x) - 1), x),
        4: lambda x: 1.0507 * np.where(x > 0, x, 1.6732 * (np.exp(x) - 1)),
        5: lambda x: 1 / (1 + np.exp(-x))
    }

    derivatives = {
        1: lambda x: np.greater(x, 0).astype(np.float64),
        2: lambda x: np.where(x > 0, 1, 0.05),
        3: lambda x: np.where(x > 0, 1, 0.1 * np.exp(x)),
        4: lambda x: np.where(x > 0, 1, selu(x) + 1 - selu(x)),
        5: lambda x: sigmoid(x) * (1 - sigmoid(x))
    }

    if deriv is None:
        return functions[param](x)
    else:
        return derivatives[param](x) 

def sort_data_random(X,Y):
    indices_aleatorios = np.random.permutation(X.shape[1])
    # Obtener los datos reordenados de X e Y
    X_r = X[:,indices_aleatorios]
    Y_r = Y[:,indices_aleatorios]
    return X_r,Y_r

def iniWs(N,hn):
    W=[]
    W=iniW(hn,N)
    V=iniW(N,hn)
    return(W,V)

# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
# STEP 1: Feed-forward of AE
def ae_forward(x,w1,v,param):
    z1=np.matmul(w1,x)
    a1=activation_function(param.actFunc,z1)
    z2=np.matmul(v,a1)
    a2=z2
    a=[x,a1,a2]
    z=[z1,z2]
    return(z,a)      
# STEP 2: Feed-Backward for AE
def gradW(a,z,w,v,param):   
    e       = a[2]-a[0]
    Cost    = np.sum(np.sum(e**2))/(2*e.shape[1])
    gW=[]
    delta_decoder=e*1
    gw_decoder=np.matmul(delta_decoder,a[1].T)
    delta_encoder=np.matmul(v.T,delta_decoder)*activation_function(param.actFunc,z[0],1)
    gw_encoder=delta_encoder@a[0].T
    gW=[gw_encoder,gw_decoder]
    return(gW,Cost)        


def updW_rmsprop(w,v,gw,vlist,param):
    beta,eps = 0.9, 1e-8
    v_encoder=beta*vlist[0]+(1-beta)*(gw[0])**2
    gRMS_encoder=(1/np.sqrt(v_encoder+eps))*gw[0]
    peso_encoder=w-param.lr*gRMS_encoder
    v_decoder=(beta*vlist[1])+((1-beta)*(gw[1]**2))
    gRMS_decoder=(1/np.sqrt(v_decoder+eps))*gw[1]
    peso_decoder=v-param.lr*gRMS_decoder

    return(peso_encoder,peso_decoder,[v_encoder,v_decoder])
# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w,v,gw,param):
    beta, eps = 0.9, 1e-8
    vnext=beta*v+(1-beta)*(gw)**2
    gRMS=(1/np.sqrt(vnext+eps))*gw
    wnext=w-param.lr*gRMS
    return(wnext,vnext)
# Softmax's gradient
def gradW_softmax(a,y,z,w,param):
    costo=(-1/y.shape[1])*np.sum(y * np.log(a[1]))       
    gW=-(1/y.shape[1])*((y-a[1])@a[0].T)    
    return(gW,costo)
def foward_sft(x,w):
    z=w@x
    a=softmax(z)
    aoutput=[x,a]
    return z,aoutput
# Calculate Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))

def fw(x,w,param):
    z=np.matmul(w,x)
    a=activation_function(param.actFunc,z)
    return a

# save weights SAE and costo of Softmax
def save_w_dl(W,Ws,cost):    
    np.savetxt('costo.csv', np.array(cost).T, delimiter=',')
    np.savez("wSoftmax.npz", Ws=Ws)
    np.savez("wAEs.npz", *W)
    
