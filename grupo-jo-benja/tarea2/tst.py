import numpy as np
import utility as ut
import prep as prep


#load data for testing
def load_data_tst():
    #...    
    return(xv,yv)    


#load weight of the DL in numpy format
def load_w_dl():
    Vs= np.load('wSoftmax.npz')
    Ws= np.load('wAEs.npz')
    W=[]
    V=0
    for x in Ws:
        W.append(Ws[x]) 
    for x in Vs:
        V=Vs[x]
     
    return(np.array(W),np.array(V))    



# Feed-forward of the DL
def forward_dl(x,W,V,param):        
    for pesoCapaOculta in W:
        x=ut.fw(x,pesoCapaOculta,param)
    z,zv=ut.foward_sft(x,V)
    zv=zv[1]
    return(zv)


# Measure
def metricas(yv,hv):
    cm,fscore=confusion_matrix(hv,yv)     
    return(cm,fscore)
    
#Confusion matrix
def confusion_matrix(predicted,real):
    y_true = real.T
    y_pred = predicted.T

    # Calcular el número de clases diferentes
    num_classes = y_true.shape[1]

    # Crear una matriz de ceros de tamaño (num_classes, num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Calcular la matriz de confusión
    for i in range(y_true.shape[0]):
        cm[np.argmax(y_true[i]), np.argmax(y_pred[i])] += 1

    precision = np.diag(cm) / cm.sum(axis=0)

    # Recall = TP / (TP + FN)
    recall = np.diag(cm) / cm.sum(axis=1)

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    fscore = 2 * (precision * recall) / (precision + recall)
    fscore = np.append(fscore, np.mean(fscore))
    return (cm,fscore)

def save_measure(cm,Fsc):
    np.savetxt('cm.csv', cm, delimiter=',')
    np.savetxt('Fsc.csv', Fsc.T, delimiter=',')
    
    return()


# Beginning ...
def main():
    p_sae,p_sft = prep.load_config()		
    trsh,trsh2,xv,yv  = prep.load_data_csv()
    W,V      = load_w_dl()
    zv     = forward_dl(xv,W,V,p_sae)      		
    cm,Fsc = metricas(yv,zv)
    save_measure(cm,Fsc)		
    print(Fsc*100)
    print('Fsc-mean {:.5f}'.format(Fsc.mean()*100))
	

if __name__ == '__main__':   
	 main()

