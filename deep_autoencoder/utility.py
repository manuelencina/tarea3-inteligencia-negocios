import numpy  as np

# Configuration of the SAEs
def load_config():
    cnf_dae     = np.loadtxt("cnf_dae.csv", delimiter = ",")
    cnf_softmax = np.loadtxt("cnf_softmax.csv")

    class ParamSae:
        nclasses    = int(cnf_dae[0])
        nframe      = int(cnf_dae[1])
        frame_size  = int(cnf_dae[2])
        p_training  = cnf_dae[3]
        encoder_act = int(cnf_dae[4])
        max_iter    = int(cnf_dae[5])
        batch_size  = int(cnf_dae[6])
        lr          = cnf_dae[7]
        encoders    = cnf_dae[8:]

    class ParamSft:
        max_iter     = int(cnf_softmax[0])
        lr           = cnf_softmax[1]
        batch_size   = int(cnf_softmax[2])

    params_sae = ParamSae()
    params_sft = ParamSft()

    return (params_sae, params_sft)

# Initialize weights of DAE 
def iniWs(x,encoders):
    W_encoder = []
    W_decoder = []
    encoders = encoders.astype(int)
    W_encoder.append(iniW(encoders[0],x.shape[0]))
    W_decoder.insert(0, iniW(x.shape[0], encoders[0]))
    for i in range(1, len(encoders)):
        W_encoder.append(iniW(encoders[i], encoders[i-1]))
        W_decoder.insert(0, iniW(encoders[i-1], encoders[i]))

    W = []
    W.extend(W_encoder)
    W.extend(W_decoder)
    return W

# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r  
    return(w)

def sort_data_random(X,Y):
    indices_aleatorios = np.random.permutation(X.shape[1])
    # Obtener los datos reordenados de X e Y
    X_r = X[:,indices_aleatorios]
    Y_r = Y[:,indices_aleatorios]
    return X_r,Y_r


def dae_forward(xe,w,param):
    a=[]
    z=[]
    z.append(np.matmul(w[0],xe))
    a.append(activation_function(param.encoder_act,z[0]))
    for i in range(1,len(w)):
        z.append(np.matmul(w[i],a[-1]))
        a.append(activation_function(param.encoder_act,z[-1]))
    return z,a
# STEP 1: Feed-forward of AE
# def dae_forward(x,...):
#     ...
#     return()    


#Activation function
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
# STEP 2: Feed-Backward for DAE
def gradW(a,w2):   

    return()        

# Update DAE's weight via mAdam
# def updW_madam():
#         ...    
#     return(...v)
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
def save_w_dl():    
    pass
    
