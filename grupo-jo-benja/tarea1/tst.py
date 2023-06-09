import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm,Fsc):
    np.savetxt('cm.csv', cm, delimiter=',')
    np.savetxt('Fsc.csv', Fsc.T, delimiter=',')
    
    return()

def load_w():
    Ws= np.load('w_snn.npz')
    pesos=[]
    V=0
    W=[]
    for x in Ws:
        pesos.append(Ws[x])
    for i in range(len(pesos)):
        if i==len(pesos)-1:
             V=pesos[i]
             break
        W.append(pesos[i])
    #print(W)
        
    return(W,V)



def load_data_test(nClasses):
    x=np.loadtxt(open(f"dtest.csv", "rb"), delimiter=",")
    xv=x[:,:x.shape[1]-nClasses]
    yv=x[:,x.shape[1]-nClasses:]

    return(xv.T,yv.T)
    

# Beginning ...
def main():
	Param  =	ut.load_cnf() 			
	xv,yv  = load_data_test(Param.nClasses)
	#print(xv.shape)
	W,V     = load_w()
	zv,hv     = ut.forward(xv,W,V,Param.hiddenActFunc)
	hv=hv[-1]
	cm,Fsc = ut.metricas(yv,hv) 	
	save_measure(cm,Fsc)
		

if __name__ == '__main__':   
	 main()

