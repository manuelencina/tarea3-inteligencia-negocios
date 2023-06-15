#Training DL via mAdam
import numpy      as np
import utility    as ut


# Training miniBatch 
def train_sft_batch(x,y,W,V,S,param):
    costo = []    
    #for i in range(numBatch):   
        
    return(W,V,costo)

# Softmax's training via mAdam
def train_softmax(x,y,param):
    W,V,S    = ut.iniW(...)    
    ...    
    for Iter in range(1,par1[0]):        
        idx   = np.random.permutation(x.shape[1])
        xe,ye = x[:,idx],y[:,idx]         
        W,V,c = train_sft_batch(xe,ye,W,V,param)
        ...
               
    #return(W,Costo)    
 
# Training by using miniBatch
def train_dae_batch(x,W,param):
    M=param.batch_size
    nBatch=int(x.shape[1]/M)
    batchesX = np.array_split(x.T,nBatch)
    costo_batches=[]
    V=[]
    for w in W:
        V.append(np.zeros(w.shape)) 
    for i in range(nBatch):
        xe=batchesX[i].T
        z,a=ut.dae_forward(xe,W, param)               
        gW,costo=ut.gradW(a,z,W,v,param)
        costo_batches.append(costo)
        w,v,vlist=ut.updW_rmsprop(w,v,gW,vlist,param)


    return(w,v,costo_batches)

# DAE
def train_dae(x,y,param):       
    Ws=ut.iniWs(x,param.encoders)
    costo_prom_iter=[]
    for i in range(param.max_iter):       
        xe,ye     = ut.sort_data_random(x,y)            
        Ws,v,cList = train_dae_batch(xe,Ws,param)
        costo_iter=np.mean(cList)
        costo_prom_iter.append(costo_iter)
        if np.mod(i,10)==0:
            print("costo(Iter): ",costo_iter) 
    return(w1) 



#load Data for Training
def load_data_trn():
  x_train=np.loadtxt(open("X_train.csv", "rb"), delimiter=",")
  y_train=np.loadtxt(open("Y_train.csv", "rb"), delimiter=",")

  return(x_train.T,y_train.T) 

# Configuration of the DAE
def load_cnf_dae():      
    par = np.genfromtxt('cnf_dae.csv',delimiter=',')    
    ...
    return(...)



# Beginning ...
def main():
    p_dae,p_sft = ut.load_config()          
    xe,ye       = load_data_trn()   
    W,Xr        = train_dae(xe,ye,p_dae)         
    Ws, cost    = train_softmax(Xr,ye,...)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

