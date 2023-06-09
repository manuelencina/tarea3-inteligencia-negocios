# SNN's Training :

import pandas     as pd
import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse():
    
    return



#miniBatch-SGDM's Training 
def trn_minibatch(W,V,x,y,param):
    M=param.batchSize
    nBatch=int(x.shape[0]/M)
    costosBatches=[]
    batchesX = np.array_split(x,nBatch)
    #print(len(batchesX))
    batchesY = np.array_split(y,nBatch)
    v1=np.zeros(W[0].shape)
    vs=np.zeros(V.shape)
    if param.hiddenNode2!=0:
         v2=np.zeros(W[1].shape)
         vlist=[v1,v2,vs]
    else: vlist=[v1,vs]
    
    for i in range(nBatch):
        xe=batchesX[i].T
        ye=batchesY[i].T
        z,h=ut.forward(xe,W,V,param.hiddenActFunc)
        costo,gW=ut.gradW(xe,z,h,W,V,ye,param)
        costosBatches.append(costo)
        W,V,vlist=ut.updWV_sgdm(W,V,gW,param,vlist)

    return(W,V,costosBatches)

#SNN's Training 
def train(x,y,Param):    
    W,V   = ut.iniWs(x.shape[1],Param)
    MSE=[]
    for i in range(Param.maxIter):
        X,Y= ut.sort_data_random(x,y)
        W,V,Clist=trn_minibatch(W,V,X,Y,Param)
        costo_iter=np.mean(Clist)
        MSE.append(costo_iter)
        if np.mod(i,10)==0:
            print("MSE(Iter): ",costo_iter)
  
    
    return(W,V,MSE)

# Load data to train the SNN
def load_data_trn(nClasses):
    x=np.loadtxt(open(f"dtrain.csv", "rb"), delimiter=",")
    xe=x[:,:x.shape[1]-nClasses]
    ye=x[:,x.shape[1]-nClasses:]
    
    return(xe,ye)
    
   
# Beginning ...
def main():
    Param       = ut.load_cnf()            
    xe,ye       = load_data_trn(Param.nClasses)   
    W,V,Cost      = train(xe,ye,Param)             
    ut.save_w_cost(W,V,Cost)
       
if __name__ == '__main__':   
	 main()

