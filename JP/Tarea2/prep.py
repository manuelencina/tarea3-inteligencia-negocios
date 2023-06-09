import numpy      as np

# Save Data : training and testing
def save_data(data_train, data_test):

  # Separando las clases de los datos...
  xe = data_train[:, :-1]
  ye = data_train[:, -1]
  
  xv = data_test[:, :-1]
  yv = data_test[:, -1]

  # Asignandole label binaria a las clases
  ye = binary_label(ye.astype(int))
  yv = binary_label(yv.astype(int))

  # Guardando los datos por separado
  np.savetxt("Xe.csv", xe.T, delimiter=',')
  np.savetxt("Ye.csv", ye.T, delimiter=',', fmt='%d')

  np.savetxt("Xv.csv", xv.T, delimiter=',')
  np.savetxt("Yv.csv", yv.T, delimiter=',', fmt='%d')

  return


# Binary Label Transofma las clases del arreglo y en representaciones binarias. 
# ej: 1 = [1,0,0,0], 2 = [0,1,0,0] ...
def binary_label(y):

    K = np.unique(y).shape[0]
    res = np.eye(K)[(y-1).reshape(-1)]
    return res.reshape(list(y.shape)+[K]).astype(int)


# Load data csv
def load_data_csv():
  
  data_train = np.loadtxt("train.csv", dtype=float, delimiter=',')
  data_test = np.loadtxt("test.csv", dtype=float, delimiter=',')

  return data_train, data_test


# Beginning ...
def main():        
    DataTrn, DataTst            = load_data_csv()
    save_data(DataTrn, DataTst)
    print("Prep finalizado correctamente")
    

if __name__ == '__main__':   
	 main()

