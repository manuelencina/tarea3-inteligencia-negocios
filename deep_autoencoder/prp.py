import numpy      as np
import utility    as ut

# normalize data 
def data_norm(X):
  
  for i in range(0, len(X)):
     
     max_value = max(X[i])
     min_value = min(X[i])
     diff= max_value - min_value
     if diff != 0:
      X[i] = normalize_vector(X[i])

  return X

#Esta funcion normaliza un vector cualquiera para que sus valores estén en el rango de 0,99 y 0,01
def normalize_vector(X):
  
   diff= max(X) - min(X)
   if diff != 0:
    X = ( (X-min(X)) / diff ) *(0.98) + 0.01

   return X

# Crate new Data : Input and Label 
def create_input_label(data: dict, params_sae: object):
  nframe      = params_sae.nframe
  frame_size  = params_sae.frame_size
  nclasses    = params_sae.nclasses
  Y           = np.array([])
  X           = np.array([])

  for idx, s in enumerate(data.values()):
    signals     = s.T[:, :nframe * frame_size].reshape(s.T.shape[0], nframe, frame_size)
    ft          = np.fft.fft(signals, axis=2)
    amplitudes  = np.abs(ft[:, :, :ft.shape[2]//2]).reshape(-1, ft.shape[2]//2)
    Y_binary    = binary_label(idx, amplitudes.shape[0], nclasses)
    Y           = stack_arrays(Y, Y_binary)
    X           = stack_arrays(X, amplitudes)

  return X, Y

    
def stack_arrays(arr: np.ndarray, new_arr: np.ndarray) -> np.ndarray:
  return np.concatenate((arr, new_arr)) if arr.shape[0] != 0 else new_arr

# Save Data : training and testing
def save_data_csv(X: np.ndarray, Y: np.ndarray, sae_params: object):
  p_training                        = sae_params.p_training
  X_train, Y_train, X_test, Y_test  = create_dtrn_dtst(X, Y, p_training)
  np.savetxt("X_train.csv", X_train, delimiter=",", fmt="%.4f")
  np.savetxt("X_test.csv", X_test, delimiter=",", fmt="%.4f")
  np.savetxt("Y_train.csv", Y_train, delimiter=",", fmt="%.4f")
  np.savetxt("Y_test.csv", Y_test, delimiter=",", fmt="%.4f")


def create_dtrn_dtst(X: np.ndarray, Y: np.ndarray, p: float):
  M = np.concatenate((X, Y), axis=1)
  np.random.shuffle(M)
  split_index       = int(M.shape[0] * p)
  trn_set, test_set = M[:split_index, :], M[split_index:, :]
  X_train, Y_train  = trn_set[:, : X.shape[1]], trn_set[:, -Y.shape[1]:]
  X_test, Y_test    = test_set[:, : X.shape[1]], test_set[:, -Y.shape[1]:]
  return X_train, Y_train, X_test, Y_test

# Binary Label
def binary_label(i: int, m: int, n: int):
  binary_array       = np.zeros((m, n))
  binary_array[:, i] = 1
  return binary_array


# Load data csv
def load_class_csv(n_classes: int) -> dict:
  return {
      f"class{i}": np.loadtxt(
          f'DATA/class{i}.csv', delimiter=',')
      for i in range(1, n_classes + 1)
  }


# Configuration of the DAEs
def load_cnf_dae():      
  cnf_dae, _ = ut.load_config()
  return cnf_dae


# Beginning ...
def main():
  sae_params      = load_cnf_dae()
  data            = load_class_csv(sae_params.nclasses)
  X, Y            = create_input_label(data, sae_params)
  X_norm = data_norm(X)
  save_data_csv(X_norm, Y, sae_params)
    

if __name__ == '__main__': 
	 main()


