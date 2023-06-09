# SNN's Training :

import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_and_cost(W, Cost):
    
    np.savetxt("costo.csv", Cost, delimiter=',')

    W = W[1:]
    np.savez("w_snn.npz", *W)
    return

#miniBatch-SGDM's Training 
def trn_minibatch(X,Y,W,V,Param):

    N = X.shape[1]
    M = Param["miniBatch_size"]
    nBatch = int(N/M)
    lb = 0
    ub = M 
    Cost = []


    for i in range(0, nBatch):
        
        xe = X[: , lb:ub]
        ye = Y[:, lb:ub]
        lb = lb + M
        ub = ub + M
        Act, Z = ut.forward(xe, W, Param)
        gW, cost_i = ut.gradW(Act, Z, ye, W, Param)
        Cost.append(cost_i)
        W, V = ut.updWV_sgdm(W, V, gW, Param)
    
    return Cost, W, V

#SNN's Training 
def train(X,Y,Param): 
 
    W,V   = ut.iniWs(Param)

    max_iter = Param["max_iter"]
    
    mse = np.repeat(0.0, max_iter)
    for i in range(0, max_iter):
        X_t,Y_bt = ut.sort_data_random(X, Y)
        Cost, W, V = trn_minibatch(X_t,Y_bt,W,V,Param)
        mse[i] = np.average(Cost)
        if i%10 == 0:
             print("\n Iterar-SGD:", i, mse[i])
         
    return W, mse

# Load data to train the SNN
def load_data_trn():
    
    x_train = np.loadtxt("x-train.csv", dtype=float, delimiter=',')
    y_train = np.loadtxt("y-train.csv", dtype=int, delimiter=',')
    return x_train, y_train
    
   
# Beginning ...
def main():
    param       = ut.load_cnf()
    xe,ye       = load_data_trn()
    W,Cost      = train(xe,ye,param)
    save_w_and_cost(W,Cost)
    print("Training finalizado correctamente.")
       
if __name__ == '__main__':   
	 main()
