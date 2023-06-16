#Training DL via mAdam
import numpy      as np
import utility    as ut


# Training miniBatch 
def train_sft_batch(x,y,W,V,S,param):
    M=param.batch_size
    nBatch=int(x.shape[1]/M)
    batchesX = np.array_split(x.T,nBatch)
    batchesY = np.array_split(y.T,nBatch)    
    costo_batches=[]
    for i in range(nBatch):
        xe=batchesX[i].T
        ye=batchesY[i].T 
        z,a=ut.forward_sft(xe,W)               
        gW,costo=ut.gradW_softmax(a,ye)
        costo_batches.append(costo)
        W,V,S=ut.updW_sft_madam(W,gW,V,S,i,param.lr)
    return(W,costo_batches)  

# Softmax's training via mAdam
def train_softmax(x,y,param):
    W        = ut.iniW(y.shape[0],x.shape[0])
    V        = np.zeros(W.shape) 
    S = np.zeros(W.shape)
    costo_prom_iter=[]   
    for i in range(param.max_iter):        
        xe,ye     = ut.sort_data_random(x,y)        
        W,costo = train_sft_batch(xe,ye,W,V,S,param)
        costo_iter=np.mean(costo)
        costo_prom_iter.append(costo_iter)
        if np.mod(i,10)==0:
            print("costo_softmax(Iter): ",costo_iter) 

               
    return(W,costo_prom_iter)
               
    #return(W,Costo)    
 
# Training by using miniBatch
def train_dae_batch(x,W,param):
    M=param.batch_size
    nBatch=int(x.shape[1]/M)
    batchesX = np.array_split(x.T,nBatch)
    costo_batches=[]
    V=[]
    S=[]
    for w in W:
        V.append(np.zeros(w.shape)) 
        S.append(np.zeros(w.shape))
    for i in range(nBatch):
        xe=batchesX[i].T
        z,a=ut.dae_forward(xe,W, param.encoder_act)               
        gW,costo=ut.gradW(a,z,W,param)
        costo_batches.append(costo)
        W,V,S=ut.updW_madam(W,gW,V,S,i,param.lr)


    return(W,costo_batches)

# DAE
def train_dae(x,y,param):       
    Ws=ut.iniWs(x,param.encoders)
    costo_prom_iter=[]
    for i in range(param.max_iter):       
        xe,ye     = ut.sort_data_random(x,y)            
        Ws,cList = train_dae_batch(xe,Ws,param)
        costo_iter=np.mean(cList)
        costo_prom_iter.append(costo_iter)
        if np.mod(i,10)==0:
            print("costo(Iter): ",costo_iter)
    z,a = ut.dae_forward(x,Ws,param.encoder_act)
    Xr = a[-1]
    return(Ws, Xr) 

#load Data for Training
def load_data_trn():
  x_train=np.loadtxt(open("X_train.csv", "rb"), delimiter=",")
  y_train=np.loadtxt(open("Y_train.csv", "rb"), delimiter=",")

  return(x_train.T,y_train.T) 


# Beginning ...
def main():
    p_dae,p_sft = ut.load_config()          
    xe,ye       = load_data_trn()   
    W, Xr       = train_dae(xe,ye,p_dae)         
    Ws, cost    = train_softmax(Xr,ye,p_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

