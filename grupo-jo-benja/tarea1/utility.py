# My Utility : auxiliars functions

import pandas as pd
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
     
#load parameters to train the SNN
def sort_data_random(X,Y):
    indices_aleatorios = np.random.permutation(X.shape[0])
    # Obtener los datos reordenados de X e Y
    X_r = X[indices_aleatorios]
    Y_r = Y[indices_aleatorios]
    return X_r,Y_r
def load_cnf():
    aux=np.loadtxt('cnf.csv')
    class Param:
        nClasses=int(aux[0])
        nFrame=int(aux[1])
        sizeFrame=int(aux[2])
        descoLevel=int(aux[3])
        hiddenNode1=int(aux[4])
        hiddenNode2=int(aux[5])
        hiddenActFunc=int(aux[6])
        trainSize=aux[7]
        batchSize=int(aux[8])
        learningRate=aux[9]
        betaCoef=aux[10]
        maxIter=int(aux[11])
    Params=Param()
    #print(Params.trainSize)
    return(Params)


# Initialize weights for SNN-SGDM
def iniWs(N,Param):
    hd1=Param.hiddenNode1
    hd2=Param.hiddenNode2
    W=[]
    W.append(iniW(hd1,N))
    if hd2!=0:
        W.append(iniW(hd2,hd1))
        V=iniW(Param.nClasses,hd2)
    else:
        V=iniW(Param.nClasses,hd1)

    return(W,V)

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of SNN
def forward(x,W,V,actFunc):
    z=[]
    h=[]
    #print(x)
    #z=Multiplicacion de matrices
    #H=matriz activada con funcion
    #print(W[0].shape,x.shape)
    z1=np.matmul(W[0],x)
    z.append(z1)
    h1=activation_function(actFunc,z1)
    h.append(h1)
    if(len(W)!=1):
        z2=np.matmul(W[1],h1)
        z.append(z2)
        h2=activation_function(actFunc,z2)
        h.append(h2)
        z3=np.matmul(V,h2)
        z.append(z3)
    else:
        z3=np.matmul(V,h1)
        z.append(z3)
    h3=activation_function(5,z3)
    h.append(h3)
   
    return(z,h) 


#Feed-Backward of SNN
def gradW(x,z,h,W,V,Y,param):
    E=0
    capas=0
    gW=[]
    deltas=[]
    if param.hiddenNode2==0:
        capas=1
    else:
        capas=2
    hy=h[-1]
    C=0
    for i in range(Y.shape[0]):
        C=C+np.sum(((hy[i]-Y[i])**2))/2
    C=C/Y.shape[0]

    deltaSalida=(h[capas]-Y)*activation_function(5,z[capas],True)
    #print(deltaSalida.shape)
    gSalida=deltaSalida@(h[capas-1].T) #GRADIENTE SALIDA
    gW.append(gSalida)
    i=capas
    #print(gSalida.shape)
    #GRADIENTES
    while i!=0:
        if i==capas:
            deltaOculta=(V.T@deltaSalida)*activation_function(param.hiddenActFunc,z[i-1],True)
            if i-2<0:
                gOculta=deltaOculta@x.T
            else:
                gOculta=deltaOculta@h[i-2].T
            gW.append(gOculta)
            deltas.append(deltaOculta)
            i=i-1
            continue
        #print(gW[-1].shape)
        deltaOculta=(W[i].T@deltas[-1])*activation_function(param.hiddenActFunc,z[i-1],True)
        gOculta=deltaOculta@x.T
        gW.append(gOculta)
        i=i-1
        
    return(C,gW)    

# Update W and V
def updWV_sgdm(W,V,gW,param,vlist):
    #Ajuste peso salida
    vlist[-1]=param.betaCoef*vlist[-1]+param.learningRate*gW[0]#Momentum capa salida
    V=V-vlist[-1]
    #Ajuste peso de nodos ocultos
    #print(gW[0].shape,gW[1].shape)
    for i in range(len(gW)-1):
        #print(gW[i].shape,vlist[i].shape)
        vlist[i]=param.betaCoef*vlist[i]+param.learningRate*np.flip(gW)[i]#Momentum capa oculta i
        W[i]=W[i]-vlist[i]   
    return(W,V,vlist)

# Measure
def metricas(yv,hv):
    cm,fscore=confusion_matrix(hv,yv)     
    return(cm,fscore)
    
#Confusion matrix
def confusion_matrix(predicted,real):
    y_true = real.T
    y_pred = predicted.T

    # Calcular el número de clases diferentes
    num_classes = y_true.shape[1]

    # Crear una matriz de ceros de tamaño (num_classes, num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Calcular la matriz de confusión
    for i in range(y_true.shape[0]):
        cm[np.argmax(y_true[i]), np.argmax(y_pred[i])] += 1

    precision = np.diag(cm) / cm.sum(axis=0)

    # Recall = TP / (TP + FN)
    recall = np.diag(cm) / cm.sum(axis=1)

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    fscore = 2 * (precision * recall) / (precision + recall)
    fscore = np.append(fscore, np.mean(fscore))
    return (cm,fscore)

def save_w_cost(W,V,cost):
    np.savetxt('costo.csv', np.array(cost).T, delimiter=',')
    if len(W)!=1:
        np.savez("w_snn.npz", w1=W[0], w2=W[1],v=V)
    else:
        np.savez("w_snn.npz", w1=W[0],v=V)
#-----------------------------------------------------------------------
