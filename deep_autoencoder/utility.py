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


def dae_forward(xe,w,act):
    a=[xe]
    z=[]
    z.append(np.matmul(w[0],xe))
    a.append(activation_function(act,z[0]))
    for i in range(1,len(w)):
        z.append(np.matmul(w[i],a[-1]))
        a.append(activation_function(act,z[-1]))
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
def gradW(a,z,w,param):   
    e       = a[-1]-a[0]
    Cost    = np.sum(np.sum(e**2))/(2*e.shape[1])
    gW=[]
    aux_w = w.copy()
    aux_w.reverse()
    aux_a = a.copy()
    aux_a.reverse()
    aux_z = z.copy()
    aux_z.reverse()
    delta=e*activation_function(param.encoder_act,z[-1],True)
    gW.append(np.matmul(delta,a[-2].T))
    for i in range(1,len(aux_a)-1):
        delta=np.matmul(aux_w[i-1].T,delta)*activation_function(param.encoder_act,aux_z[i],True)
        gW.append(delta@aux_a[i+1].T)

    return(gW,Cost)           

# Update DAE's weight via mAdam
def updW_madam(W,gW,V,S,t,mu):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    gW.reverse()
    for i in range(len(gW)):
        V[i] = beta1*V[i] + (1-beta1)*gW[i]
        S[i] = beta2*S[i] + (1-beta2)*(gW[i]**2)
        gAdam = np.sqrt(1-(beta2**t))/(1-(beta1**t) +0.000000000000000000001)
        gAdam = gAdam*(V[i]/(np.sqrt(S[i])+ eps) )
        W[i] = W[i] - mu*gAdam
    return W,V,S
# Update Softmax's weight via mAdam
def updW_sft_madam(W,gW,V,S,t,mu):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    V = beta1*V + (1-beta1)*gW
    S = beta2*S + (1-beta2)*(gW**2)
    gAdam = np.sqrt(1-(beta2**t))/(1-(beta1**t) +0.000000000000000000001)
    gAdam = gAdam*(V/(np.sqrt(S)+ eps) )
    W = W - mu*gAdam
    return W,V,S

# Softmax's gradient
def gradW_softmax(a,y):
    costo=(-1/y.shape[1])*np.sum(y * np.log(a[1]))       
    gW=-(1/y.shape[1])*((y-a[1])@a[0].T)    
    return(gW,costo)

# Calculate Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))

def forward_sft(x,w):
    z=w@x
    a=softmax(z)
    aoutput=[x,a]
    return z,aoutput

# save weights SAE and costo of Softmax
def save_w_dl(W,Ws,cost):    
    np.savetxt('costo.csv', np.array(cost).T, delimiter=',')
    W.append(Ws)
    np.savez("wdae.npz", *W)
    
