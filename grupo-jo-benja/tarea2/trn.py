#Training DL via RMSProp+Pinv

import numpy      as np
import utility    as ut
import prep as prep

# Training miniBatch for softmax
def train_sft_batch(x,y,w,vm,param):
    M=param.batchSize
    nBatch=int(x.shape[1]/M)
    batchesX = np.array_split(x.T,nBatch)
    batchesY = np.array_split(y.T,nBatch)    
    costo_batches=[]
    for i in range(nBatch):
        xe=batchesX[i].T
        ye=batchesY[i].T 
        z,a=ut.foward_sft(xe,w)               
        gW,costo=ut.gradW_softmax(a,ye,z,w,param)
        costo_batches.append(costo)
        w,vm=ut.updW_sft_rmsprop(w,vm,gW,param)
    return(w,costo_batches)
# Softmax's training via SGD with Momentum
def train_softmax(x,y,p_sft,p_sae):
    W        = ut.iniW(y.shape[0],x.shape[0])
    V        = np.zeros(W.shape) 
    costo_prom_iter=[]   
    for i in range(p_sft.maxIter):        
        xe,ye     = ut.sort_data_random(x,y)        
        W,costo = train_sft_batch(xe,ye,W,V,p_sft)
        costo_iter=np.mean(costo)
        costo_prom_iter.append(costo_iter)
        if np.mod(i,10)==0:
            print("costo_softmax(Iter): ",costo_iter) 

               
    return(W,costo_prom_iter)    
 
# AE's Training with miniBatch
def train_ae_batch(x,y,w,v,param):
    M=param.batchSize
    nBatch=int(x.shape[1]/M)
    batchesX = np.array_split(x.T,nBatch)
    batchesY = np.array_split(y.T,nBatch)    
    costo_batches=[]
    v1=np.zeros(w.shape)
    vs=np.zeros(v.shape)
    vlist=[v1,vs] 
    for i in range(nBatch):
        xe=batchesX[i].T
        #ye=batchesY[i].T 
        z,a=ut.ae_forward(xe,w,v,param)               
        gW,costo=ut.gradW(a,z,w,v,param)
        costo_batches.append(costo)
        w,v,vlist=ut.updW_rmsprop(w,v,gW,vlist,param)


    return(w,v,costo_batches)
# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x,y,encoder_index,param):       
    w1,v=ut.iniWs(x.shape[0],param.nEncoders[encoder_index].astype(int))
    costo_prom_iter=[]
    #....    
    for i in range(param.maxIter):       
        xe,ye     = ut.sort_data_random(x,y) 
        #print(w1)             
        w1,v,cList = train_ae_batch(xe,ye,w1,v,param)
        costo_iter=np.mean(cList)
        costo_prom_iter.append(costo_iter)
        if np.mod(i,10)==0:
            print("costo(Iter): ",costo_iter) 
    return(w1) 
#SAE's Training 
def train_sae(x,y,param):
    W=[]
    #print(x.shape)
    for i in range(len(param.nEncoders)):        
        w1       = train_ae(x,y,i,param)   
        x        = ut.fw(x,w1,param)
        W.append(w1)                      
    return(W,x) 

#load Data for Training
def load_data_trn():
    xe,ye,n,nn=prep.load_data_csv()   
    return(xe,ye)    

# Beginning ...
def main():
    p_sae,p_sft = prep.load_config()           
    xe,ye       = load_data_trn()   
    W,Xr        = train_sae(xe,ye,p_sae)      
    Ws, cost    = train_softmax(Xr,ye,p_sft,p_sae)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

