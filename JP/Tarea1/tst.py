import numpy as np
import utility as ut


def save_measure(cm,Fsc):
    
    np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')

    np.savetxt("fscores.csv", np.array(Fsc))
    return

def load_w():
    data = np.load("w_snn.npz")
    lst = data.files

    W = [None]
    for item in lst:
        W.append(data[item])

    

    return W

def load_data_test():
    x_test= np.loadtxt("x-test.csv", dtype=float, delimiter=',')
    y_test = np.loadtxt("y-test.csv", dtype=int, delimiter=',')

    y_test = ut.binary_label(y_test.astype(int))
    return x_test.T, y_test.T
    

# Beginning ...
def main():			
	xv,yv  = load_data_test()
	W      = load_w()
	A, Z     = ut.forward(xv,W, Param=ut.load_cnf())      		
	cm,Fsc = ut.metricas(yv,A[-1]) 	
	save_measure(cm,Fsc)
	print("Test finalizado correctamente.")

if __name__ == '__main__':   
	 main()
