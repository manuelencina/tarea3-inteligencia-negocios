# My Utility : auxiliars functions

import numpy  as np

# Configuration of the SAEs
def load_config():
        
    f = open("cnf_sae.csv", "r")

    lista = f.read().splitlines()
    par_sae = {
     "p_inverse": int(lista[0]),
     "activation_function": int(lista[1]),
     "max_iter": int(lista[2]),
     "miniBatch_size": int(lista[3]),
     "learning_rate": float(lista[4]),
    }

    for i in range(5, len(lista)):
        if lista[i] == '':
            i = i-1
            break
        par_sae["nodes_AE"+ str(i-5)] = int(lista[i])
    
    par_sae["num_AE"] = int(i-4)
    f.close()  

    f = open("cnf_softmax.csv", "r")

    par_sft = {
     "max_iter": int(f.readline()),
     "learning_rate": float(f.readline()),
     "miniBatch_size": int(f.readline()),
    }
  
    f.close()

    return(par_sae,par_sft)

#Initialize weights
def iniWs(iter, param):    
    

    hidden_nodes = param["nodes_AE"+str(iter)]

    W = [None, iniW(hidden_nodes, param["d_chars"])]
    V = [None, np.full(shape=(hidden_nodes, param["d_chars"]),fill_value=0)]
    
    W.append( iniW(param["d_chars"], hidden_nodes))
    V.append(np.full(shape= (param["d_chars"], hidden_nodes) ,fill_value=0))

    return(W,V)


# Initialize one-wieght
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
# STEP 1: Feed-forward of AE
def ae_forward(x,w,param):

    A = [x, None, None]
    Z = [None, None, None]
    functionNumber = param["activation_function"]

    for l in range(1, 3):
        Z[l] = np.matmul(w[l], A[l-1])

        if l == 2:
            A[l] = np.copy(Z[l])
            break
        A[l] = act_function( functionNumber , Z[l])
    
    return A, Z
#Activation functions
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
# STEP 2: Feed-Backward for AE
def gradW(a,z,w, param):   
    e       = a[2]-a[0]
    Cost    = np.sum(np.sum(e**2))/(2*e.shape[1])

    # Como la función de activación del decored es lineal
    # entonces solo se multiplica e*z2
    delta2 =np.full(shape = z[2].shape, fill_value=e)

    # Cálculo del gradiente del encoder
    gW1 = np.matmul(w[2].T, delta2) * deriva_act( param["activation_function"],z[1])
    gW1 = np.matmul(gW1, a[0].T)

    # Cálculo del gradiente del decoder
    gW2 = np.matmul(delta2, a[1].T)

    gW = [None, gW1, gW2]
    return(gW,Cost)        

# Update AE's weight via RMSprop
def updW1_rmsprop(w,v,gw,mu):
    beta,eps = 0.9, 1e-8

    for i in range(2,0,-1):
        v[i] = ( beta * v[i] ) + ((1-beta) * (gw[i]**2))
        gRMS = 1/np.sqrt(v[i]+eps) * gw[i]
        w[i] = w[i] - mu*gRMS
        
    return w,v
# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w,v,gw,mu):
    beta, eps = 0.9, 1e-8

    v = ( v * beta) + ( (gw**2) * (1-beta)  )

    gRMS = (1/np.sqrt(v +eps) ) * gw

    w = w - (gRMS*mu)

    return(w,v)
# Softmax's gradient
def gradW_softmax(x,y,a):        
    ya   = y*np.log(a)

    M = y.shape[1]
    Cost = -1 * (ya.sum() / M)

    gW = -1 * (np.matmul((y-a), x.T) / M)
    
    return(gW,Cost)
# Calculate Softmax
def softmax(z):
        # z-np.max(z)
        exp_z = np.exp(z-np.max(z))
        exp_z = np.array(exp_z, dtype=np.float64)
        return(exp_z/exp_z.sum(axis=0,keepdims=True))


# save weights SAE and costo of Softmax
def save_w_dl(W,Ws,cost):    

    np.savez("wAEs.npz", *W)

    np.savez("wSoftmax.npz", Ws)
    np.savetxt("costo.csv", cost, delimiter=',')

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

# Returns confusion matrix and fscore
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