#Training DL via RMSProp+Pinv

import numpy      as np
import utility    as ut

# Training miniBatch for softmax
def train_sft_batch(x,y,W,V,param):
    costo = []
    M = param["miniBatch_size"]
    numBatch = np.int16(np.floor(x.shape[1]/M))    
    lb = 0
    ub = M
    for i in range(numBatch):   
        xe = x[:, lb:ub]
        ye = y[:, lb:ub]
        lb = lb + M
        ub = ub + M

        z = np.matmul(W,xe)
        a = ut.softmax(z)

        gWsoft, cost_i = ut.gradW_softmax(xe, ye, a)

        W, V = ut.updW_sft_rmsprop(W, V, gWsoft, param["learning_rate"])

        costo.append(cost_i)
                
    return(W,V,costo)
# Softmax's training via SGD with Momentum
def train_softmax(x,y,par2):
    W        = ut.iniW(y.shape[0],x.shape[0])
    V        = np.zeros(W.shape) 
    Costo = []
    for Iter in range(par2["max_iter"]):
        print("SOFTMAX ITERACIÓN: ", Iter+1)
        idx   = np.random.permutation(x.shape[1])
        xe,ye = x[:,idx],y[:,idx]         
        W,V,c = train_sft_batch(xe,ye,W,V,par2)
        Costo.append(np.average(c))
               
    return(W,Costo)    
 
# AE's Training with miniBatch
def train_ae_batch(x,W,V,param):

    M = param["miniBatch_size"]
    numBatch = np.int16(np.floor(x.shape[1]/M))    
    cost= [] 
    lb = 0
    ub = M
    for i in range(numBatch):                
        xe = x[:, lb:ub]
        lb = lb + M
        ub = ub + M
        A,Z = ut.ae_forward(xe, W, param)
        gW, cost_i = ut.gradW(A,Z,W, param)
        cost.append(cost_i)
        W, V = ut.updW1_rmsprop(W,V,gW,param["learning_rate"])

    return(W,V,cost)
# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, iteration, param):        
    param["d_chars"] = x.shape[0]
    w,v= ut.iniWs(iteration, param)
    for Iter in range(0,param["max_iter"]):
        print("AE: ", iteration+1, "ITERACIÓN: ", Iter+1)
        xe     = x[:,np.random.permutation(x.shape[1])]
        w,v,c = train_ae_batch(xe,w,v,param)
        
    # w = [None, encoderW, decoderW]
    # Only need the encoder weights
    return(w[1]) 
#SAE's Training 
def train_sae(x,param):
    W=[]
    for i in range(param["num_AE"]):        
        w1       = train_ae(x,i,param)
        x        = ut.act_function(param["activation_function"], np.matmul(w1,x))
        W.append(w1)
    return(W,x) 

#load Data for Training
def load_data_trn():
    
    xe = np.loadtxt("Xe.csv", dtype='float64', delimiter=',')
    ye = np.loadtxt("Ye.csv", dtype='float64', delimiter=',')

    return(xe,ye)    

# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()           
    xe,ye       = load_data_trn()   
    W,Xr        = train_sae(xe,p_sae)         
    Ws, cost    = train_softmax(Xr,ye,p_sft)
    ut.save_w_dl(W,Ws,cost)
    print("Training finalizado correctamente")
if __name__ == '__main__':   
	 main()
