# My Utility : auxiliars functions
import numpy  as np

#load parameters to train the SNN
def load_cnf():
    f = open("cnf.csv", "r")

    configParams = {
     "class_number": int(f.readline()),
     "frame_number": int(f.readline()),
     "frame_size": int(f.readline()),
     "decomp_level": int(f.readline()),
     "hidden_nodes1": int(f.readline()),
     "hidden_nodes2": int(f.readline()),
     "activation_function": int(f.readline()),
     "training_rate": float(f.readline()),
     "miniBatch_size": int(f.readline()),
     "learn_rate": float(f.readline()),
     "beta_coef": float(f.readline()),
     "max_iter": int(f.readline())
  }
  
    f.close()

    return configParams


#Transofma las clases del arreglo y en representaciones binarias. ej: 1 = [1,0,0,0], 2 = [0,1,0,0] ...
def binary_label(y):

    K = np.unique(y).shape[0]
    res = np.eye(K)[(y-1).reshape(-1)]
    return res.reshape(list(y.shape)+[K]).astype(int)

def sort_data_random(x,y):

    # concat = np.concatenate((x , np.reshape(y,(1,-1))))
    # np.random.shuffle(concat)
    #Se junta la data X con sus respectivas labels 
    concat = np.concatenate((x , np.reshape(y,(-1,1))), axis=1)
    
    #Se mezcla la data
    np.random.shuffle(concat)

    x = concat[:,:-1]
    y = concat[:,-1]


    y = binary_label(y.astype(int))
    return x.T,y.T
# Initialize weights for SNN-SGDM
def iniWs(Param):    
    

    nodes_cape1 = Param["hidden_nodes1"]
    nodes_cape2 = Param["hidden_nodes2"]
    
    nClases = Param["class_number"]

    d = Param["decomp_level"] 

    W = [None, iniW(nodes_cape1, 2**(d+1) )]
    V = [None, np.full(shape= (nodes_cape1, 2**(d+1)),fill_value=0)]
    
    if nodes_cape2 != 0:

        W.append( iniW(nodes_cape2, nodes_cape1))
        V.append( np.full(shape= (nodes_cape2, nodes_cape1),fill_value=0))

        W.append( iniW(nClases, nodes_cape2))
        V.append( np.full(shape= (nClases, nodes_cape2),fill_value=0))

    else:
        W.append( iniW(nClases, nodes_cape1))
        V.append(np.full(shape= (nClases, nodes_cape1) ,fill_value=0))

    return(W,V)

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of SNN
def forward(X, W, Param):


    nodes_cape2 = Param["hidden_nodes2"]
    if nodes_cape2 != 0:
        Lcapes = 3
        Z = [None, None, None, None]
        A = [X, None, None, None]
    else:
        Lcapes = 2
        Z = [None, None, None]
        A = [X, None, None]

    functionNumber = Param["activation_function"]


    for l in range(1, Lcapes+1):
        Z[l] = np.matmul(W[l], A[l-1])

        if l == Lcapes:
            functionNumber = 5
            
        A[l] = act_function( functionNumber , Z[l])

    

    return A, Z

#Activation function
def act_function(function_number, x):
    if 1 == function_number:	# Relu
        return np.maximum(0, x)
    if 2 == function_number:	# L-Relu
        return np.maximum(0.01 * x, x)
    if 3 == function_number: 	# ELU
        return np.maximum(0.01 * (np.exp(x) - 1), x)
    if 4 == function_number:	# SELU
        return np.maximum(1.0507 * 1.6732 * (np.exp(x) - 1), 1.0507 * x)
    if 5 == function_number:	# Sigmoide
        return 1 / (1 + np.exp(-x))
    else:
        return None
# Derivatives of the activation funciton
def deriva_act(function_number, x):
    if 1 == function_number:	# Relu
        return np.greater(x, 0).astype(float)
    if 2 == function_number:	# L-Relu
        return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.01, lambda e: 1])
    if 3 == function_number: 	# ELU
        return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.1 * np.exp(e), lambda e: 1])
    if 4 == function_number:	# SELU
        return np.piecewise(x, [x <= 0, x > 0], [lambda e: 1.0507 * 1.6732 * np.exp(e), lambda e: 1.0507])
    if 5 == function_number:	# Sigmoide
        fx = act_function(5, x)
        return fx * (1 - fx)
    else:
        return None

#Feed-Backward of SNN
def gradW(Act, Z, ye, W, Params):    


    nodes_cape2 = Params["hidden_nodes2"]
    if nodes_cape2 != 0:
        Lcapes = 3   
        delta = [None, None, None, None]
        dE_dW = [None, None, None, None]
    else:
        Lcapes = 2
        delta = [None, None, None]
        dE_dW = [None, None, None]

    e = Act[Lcapes] - ye
    delta_l = e * deriva_act(5, Z[Lcapes])
    dE_dW_l = np.matmul(delta_l, np.transpose(Act[Lcapes-1]))


    delta[Lcapes] = delta_l
    dE_dW[Lcapes] = dE_dW_l

    fnumber = Params["activation_function"]
    for l in range(Lcapes-1, 0, -1):
        
        delta[l] = np.matmul(np.transpose(W[l+1]) , delta[l+1])  * deriva_act(fnumber, Z[l])
        dE_dW[l] = np.matmul(delta[l], np.transpose(Act[l-1]))

    Cost = calculate_mse(Act[Lcapes], ye)
    return dE_dW, Cost

# Update W and V
def updWV_sgdm(W, V, gW, Params):

    Lcapes = 2
    nodes_cape2 = Params["hidden_nodes2"]
    if nodes_cape2 != 0:
        Lcapes = 3

    beta = Params["beta_coef"]
    mu = Params["learn_rate"]
    for l in range(Lcapes, 0, -1):

        V[l] = beta* V[l] + mu * gW[l]
        W[l] = W[l] - V[l]

    return W, V

#
def calculate_mse(pred, y):
    N = y.shape[1]
    e = pred - y
    mse = np.sum(e**2)/N
    return mse

#Funcion encargada de calcular la presición 
def precision(i, cm):
	suma = np.sum(cm[i])

	if (suma > 0):
		prec = cm[i][i] / suma
	else:
		prec = 0
    
	return prec

#Funcion encargada de calcular el recall
def recall(j, cm):
	suma = np.sum(cm[:, j])
    
	if (suma > 0):
		rec = cm[j][j] / suma
	else: 
		rec = 0
    
	return rec

#Para calcular el fscore
def fscore(j, cm):
	numerator = precision(j, cm) * recall(j, cm)
	denominator = precision(j, cm) + recall(j, cm)

	if 0 == denominator:
		return 0
  
	fscore = 2 * (numerator / denominator) 
	return fscore

#Confusion matrix
def confusion_matrix(y_true, y_pred):
    m, N = y_true.shape
    cm = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            for n in range(N):
                if y_pred[j, n] == 1 and y_true[i, n] == 1:
                    cm[i, j] += 1
    return cm

# Measure
def metricas(y_true, y_pred):

    # Esto es para que el valor de clase con más peso quede con un 1, se repite el proceso para cada muestra
    for sample in y_pred.T:
        max_value_i = np.argmax(sample)
        sample[max_value_i] = 1
    
    # Esto es para que los demás valores de clase que no son 1 sean aproximados a 0
    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred)

    k = cm.shape[0]
    fscore_result = [0] * (k + 1)

    for j in range(k):
        fscore_result[j] = fscore(j, cm)

    fscore_result[k] = np.mean(fscore_result[:-1])

    return cm, fscore_result
#-----------------------------------------------------------------------