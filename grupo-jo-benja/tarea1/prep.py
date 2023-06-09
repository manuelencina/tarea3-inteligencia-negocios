import pandas     as pd
import numpy      as np
import utility    as ut

# Save Data from  Hankel's features
def save_data(X,Y,trainSize):
  # Obtener el número de filas de la matriz X
  n_filas = X.shape[0]
  print(n_filas)
  #print(trainSize)

  # Reordenar aleatoriamente los índices de las filas
  indices_aleatorios = np.random.permutation(n_filas)

  # Obtener los datos reordenados de X e Y
  X_r = X[indices_aleatorios]
  Y_r = Y[indices_aleatorios]

  # Calcular el número de filas que se utilizarán para entrenamiento
  p = trainSize # Porcentaje de datos que se utilizarán para entrenamiento
  n_train = int(np.round(n_filas * p))

  # Dividir X e Y en conjuntos de entrenamiento y prueba
  X_train = X_r[:n_train,:]
  Y_train = Y_r[:n_train]
  X_test = X_r[n_train:,:]
  Y_test = Y_r[n_train:]

  np.savetxt('dtrain.csv', np.column_stack((X_train, Y_train)), delimiter=',')
  np.savetxt('dtest.csv', np.column_stack((X_test, Y_test)), delimiter=',')
    
  return 0

# normalize data 
def data_norm(X):
# Calculamos los valores mínimos y máximos de la matriz
  #print(X[127])
  copia=X.transpose()
  new_list=[]
  largo=len(copia)
  for i in range(largo):
    x=copia[i].astype(float)
    maximo=max(x)
    minimo=min(x)
    a=0.01
    b=0.99
    for j in range(len(x)):
      x[j]=x[j].astype(float)
      x[j]=((x[j]-minimo)/(maximo-minimo))*(b-a)+a
    new_list.append(x)
  return np.array(new_list).T

# Binary Label
def binary_label():
  
  return 0

def shannon_entropy(c):
    large=len(c)
    #print(c)
    #print(large)
    n=int(np.sqrt(large))+1
    #print(np.sqrt(large))
    #print(round(3.88))
    intervals=np.linspace(min(c),max(c),n+1)
    p=[]
    for i in range(n-1):
        cont=0
        lb=intervals[i]
        ub=intervals[i+1]
        for j in range(large):
            if i==n-1:
                if c[j] >= lb and c[j] <= ub:
                    cont=cont+1
                continue
            if c[j] >= lb and c[j] < ub:
                cont=cont+1
                continue
        p.append(cont/large)
    entropy=0
    for prob in p:
        if prob==0:continue
        entropy=entropy-np.sum(prob * np.log2(prob))
    return entropy
# Fourier spectral entropy
def entropy_spectral(c):
  fft_data = np.fft.fft(c)
  amp_spec = np.abs(fft_data)
  amp_spec= amp_spec / np.sum(amp_spec)
  entropy = shannon_entropy(amp_spec) 
  return entropy

# Hankel-SVD
def hankel_svd():
     
  return (0) 
def get_diadic_values(matrix):
  Cn=[]
  x=0
  rows,columns=matrix.shape

  for i in range(rows):   
    for j in range(x,columns):

      if j==0:
        Cn.append(matrix[0][0])
        continue

      ai,aj=i,j
      C=0
      cont=0
      while ai != rows and aj!=-1:
          C=C+matrix[ai][aj]
          ai=ai+1
          aj=aj-1
          cont=cont+1
      Cn.append(C/cont)
    x=columns-1

  return np.array(Cn) 
         
         
         
        
       
         
        
   
   
# Hankel's features 
def hankel_features(X,nFrame,sizeFrame,descoLevel):
  #Parametros
  nSubVectors = len(X) // sizeFrame
  frames = np.array_split(X, nSubVectors)
  #print(frames)
  F=[]


  for x in range(nFrame):
    hankelMatrix=[]
    hList=[]
    hListAux=[]
    entropyList=[]
    diagSList=[]
    X=frames[x]
    N=len(X)
    L=2
    K=N-L+1
    descoLvl=descoLevel

    #Matriz de hankel
    for i in range(L):
      hankelMatrix.append(X[i:K+i])
    #print(hankelMatrix)
    H=np.array(hankelMatrix)
    hList.append(H)
    #print("Empezando el calculo del arbol de descomposición")

    for i in range(descoLvl):
      for j in range(len(hList)):
        U, S, VT = np.linalg.svd(hList[j],full_matrices=False)
        V=VT.T
        diagS=S

        for k in range(len(diagS)):
          Uh=U[:,k].reshape(-1, 1)
          Vh=V[:,k].reshape(-1, 1)
          Hn=diagS[k]*(Uh@Vh.T)
          hListAux.append(Hn)
          if i==descoLvl-1:
            cAux=get_diadic_values(Hn)
            Ux, Sx, VTx = np.linalg.svd(Hn,full_matrices=False)
            entropyList.append(entropy_spectral(cAux))
            diagSList.append(Sx[k])

      hList=hListAux
      hListAux=[]
    #print(np.array(entropyList).shape)
    #print(np.array(diagSList).shape)
    F.append(entropyList+diagSList)
    

  #print(entropyList)
  #print(len(cList))
  #print(hankelS)
  #print(np.array(F).shape)
  return np.array(F)


# Obtain j-th variables of the i-th class
def data_class(matrix,j):
  return matrix[:,j]


# Create Features from Data
def create_features(matrixList,Param,nVar):
  datF=[]
  labelList=[]
  identidad=np.eye(Param.nClasses)
  #F=hankel_features(0,Param.nFrame,Param.sizeFrame,Param.descoLevel)
  for i in range(Param.nClasses):
    matrices = [hankel_features(data_class(matrixList[i], j), Param.nFrame, Param.sizeFrame, Param.descoLevel) for j in range(nVar)]
    concat=np.vstack(matrices)
    datF.append(concat)
    #print(np.array(concat).shape)
    for k in range(concat.shape[0]):
      labelList.append(identidad[i])
  
  finalX=np.vstack(datF)
  #print(finalX.shape)
  finalY=np.array(labelList)
  #print(finalY.shape)
  
  return (finalX,finalY) 


# Load data from ClassXX.csv
def load_data(nClasses):
  classesMatrixList=[]
  for i in range(nClasses):
     x=np.loadtxt(open(f"Data{nClasses}\class{i+1}.csv", "rb"), delimiter=",")
     classesMatrixList.append(x)
  nVar=classesMatrixList[0].shape[1]
  return (classesMatrixList,nVar)


# Beginning ...
def main():        
  Param           = ut.load_cnf()	
  Data,nVar            = load_data(Param.nClasses)
  InputDat,OutDat = create_features(Data,Param,nVar)
  InputDat        = data_norm(InputDat)
  save_data(InputDat,OutDat,Param.trainSize)
  return


if __name__ == '__main__':   
	 main()


