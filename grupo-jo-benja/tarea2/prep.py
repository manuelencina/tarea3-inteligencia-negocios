import pandas     as pd
import numpy      as np
import utility    as ut

# Save Data : training and testing
def save_data():
      
  return


# Binary Label
def binary_label(data):
  identidad=np.eye(max(data[:,-1]).astype(int))
  lista_clases=data[:,-1].astype(int)
  clases_binarias=[]
  for i in range(len(lista_clases)):
    clases_binarias.append(identidad[lista_clases[i]-1])
  clases_binarias=np.array(clases_binarias)
  #data_with_binary_classes=np.column_stack((data[:,:-1],clases_binarias))
  return(clases_binarias)



# Load data csv
def load_data_csv():
  data_train=np.loadtxt(open("train.csv", "rb"), delimiter=",")
  data_test=np.loadtxt(open("test.csv", "rb"), delimiter=",")
  y_train=binary_label(data_train).T
  y_test=binary_label(data_test).T
  x_train=data_train[:,:-1].T
  x_test=data_test[:,:-1].T
   
  return(x_train,y_train,x_test,y_test) 


# Configuration of the SAEs
def load_config():      
    
    aux1=np.loadtxt('cnf_sae.csv')
    aux2=np.loadtxt('cnf_softmax.csv')
    class ParamSae:
        pinv=int(aux1[0])
        actFunc=int(aux1[1])
        maxIter=int(aux1[2])
        batchSize=int(aux1[3])
        lr=aux1[4]
        nEncoders=aux1[5:]

    class ParamSft:
        maxIter=int(aux2[0])
        lr=aux2[1]
        batchSize=int(aux2[2])

    params_sae=ParamSae()
    params_sft=ParamSft()
    return(params_sae,params_sft)



# Beginning ...
def main():        
    param_sae,param_sft = load_config()	
    x_train,y_train,x_test,y_test = load_data_csv()
    save_data()
    

if __name__ == '__main__':   
	 main()


