import numpy      as np
import utility    as ut


#Create x-train.csv , y-train.csv , x-test.csv and y-test.csv files with pre-procesed data, based on a trainingRate ratio for training
def create_dtrn_dtst(X,Y, trainingRate):
   
   #Se junta la data X con sus respectivas labels 
   concat = np.concatenate((X , np.reshape(Y,(-1,1))), axis=1)
   
   #Se mezcla la data
   np.random.shuffle(concat)

   totalData = len(concat)

   #Se separan los datos de training y de test
   dataTraining = concat[:int(totalData*trainingRate)]
   dataTest = concat[int(totalData*trainingRate) :]

   x_train = dataTraining[:, :-1]
   y_train = dataTraining[: , -1]

   np.savetxt("x-train.csv", x_train, delimiter=',')
   np.savetxt("y-train.csv", y_train, delimiter=',', fmt='%d')

   x_test = dataTest[: , :-1]
   y_test = dataTest[: , -1]

   np.savetxt("x-test.csv", x_test, delimiter=',')
   np.savetxt("y-test.csv", y_test, delimiter=',', fmt='%d')

   return


# normalize data 
def data_norm(X):
  
  X= X.T
  for i in range(0, len(X)):
     
     max_value = max(X[i])
     min_value = min(X[i])
     diff= max_value - min_value
     if diff != 0:
      X[i] = normalize_vector(X[i])

  X= X.T
  return X

#Esta funcion normaliza un vector cualquiera para que sus valores estén en el rango de 0,99 y 0,01
def normalize_vector(X):
  
   diff= max(X) - min(X)
   if diff != 0:
    X = ( (X-min(X)) / diff ) *(0.98) + 0.01

   return X

# Fourier spectral entropy
def entropy_spectral(arrayX):

  N = len(arrayX)
  Ix = int(np.ceil(np.sqrt(N)))
  range_of_interval = (1 - 0.01) / Ix
  entropy=0
  for i in range (0, Ix):
    lb = 0.01 + range_of_interval*i
    ub = lb + range_of_interval 

    itemsInsideInterval=0
    for item in arrayX:
      if lb <= item < ub:
        itemsInsideInterval= itemsInsideInterval + 1

    if itemsInsideInterval != 0 :
      Pi = itemsInsideInterval/N
      entropy += Pi * np.log2(Pi)

  return -entropy

# Hankel-SVD
def hankel_svd():

  return 
# Hankel's features 
def hankel_features(X, Param):
  
  nFrames = Param["frame_number"]
  Lframe = Param["frame_size"]
  jLevel = Param["decomp_level"]

  hankel_features = []
  for i in range(0, nFrames):

    sValues_c = []
    totalComponents = []
    frame = X[i*Lframe : (i+1)*Lframe]

    
    h = np.stack( (frame[:Lframe-1], frame[1:]) )
    aux1 = np.append(h[0], h[1][-1])
    aux2 = np.insert(h[1], 0, h[0][0])
    current_component = np.add(aux1, aux2)/2
    totalComponents.append(current_component)

    queue = [h]
    k=1
    for j in range(0, (2**jLevel) -1):
      
      
      h_child = queue.pop(0)
      u, s, v = np.linalg.svd(h_child, full_matrices=False)
      sValues_c.append(s)
      h1 = np.outer( u[:,0], v[0,:] ) * sValues_c[j][0]

      if k < (2**jLevel)-1:
        queue.append(h1)
        k+=1
      np.append(h1[0], (h1[1][-1])) 
      np.insert(h1[1], 0, h1[0][0])
      current_component = np.add(h1[0], h1[1])/2
      totalComponents.append(current_component)
      
      h2 = np.outer(u[:,1],v[1,:]) * sValues_c[j][1]
      if k < (2**jLevel)-1:
        queue.append(h2)
        k+=1
      np.append(h2[0], h2[1][-1])
      np.insert(h2[1], 0, h2[0][0])
      current_component = np.add(h2[0], h2[1])/2
      totalComponents.append(current_component)

    # sValues_c[2**(jLevel-1)-1:] son los valores singulares del nivel J
    # Para dejar values_of_j_level en 1d, se utiliza np.concatenate
    # queda de dimensión 2^jLevel
    values_of_j_level = np.concatenate((sValues_c[2**(jLevel-1)-1:]))

    #totalComponens[2^jLevel-1 : ] son los componentes del nivel J
    #Queda de dimensión 2^jLevel
    components_of_j_level = totalComponents[2**jLevel-1 :]

    entropys_of_j_level = []
    for c in components_of_j_level:
       fourier_c = np.fft.fft(c)
       fourier_c_amplitude = np.absolute(fourier_c)
       normalizedFourierData = normalize_vector(fourier_c_amplitude)
       entropy_c = entropy_spectral(normalizedFourierData)
       entropys_of_j_level.append(entropy_c)
    
    entropys_and_values_of_jlevel = np.concatenate((entropys_of_j_level, values_of_j_level))
    hankel_features.append(entropys_and_values_of_jlevel)

  return hankel_features

# Obtain j-th variables of the i-th class
def data_class(Data,j,i):
  
  dataClass = Data[i-1]

  return dataClass[j-1]


# Create Features from Data
def create_features(Data,Param):

  for i in range(1, Param["class_number"] +1):
    
    for j in range(1, 4+1):
      X = data_class(Data, j, i)
      F = hankel_features(X, Param)
      if j==1:
        datF=np.array(F)
      else:
        datF = np.concatenate((datF, np.array(F)))
      
    label = np.repeat(i, 4*Param["frame_number"])

    if i==1:
       Y= label
       x = np.array(datF)

    else:
      Y = np.concatenate((Y, label))
      x = np.concatenate((x, datF))

  return x, Y


# Load data from ClassXX.csv
def load_data(class_number):
  
  data_array=[]
  for i in range(1, class_number+1):
    data_file = "Data" + str(class_number) +"/class" + str(i)+".csv"
    data = np.loadtxt(data_file, dtype=float, delimiter=',')
    data_array.append(np.transpose(data))
  return data_array


# Beginning ...
def main():        
    Param           = ut.load_cnf()
    Data            = load_data(Param["class_number"])
    InputDat,OutDat = create_features(Data, Param)
    InputDat        = data_norm(InputDat)
    create_dtrn_dtst(InputDat,OutDat, Param["training_rate"])
    print("Prep finalizado correctamente")


if __name__ == '__main__':   
	 main()