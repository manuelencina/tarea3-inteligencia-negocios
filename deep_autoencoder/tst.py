import numpy as np
import utility as ut


#load data for testing
def load_data_tst():
	xv= np.loadtxt("X_test.csv", dtype=float, delimiter=',')
	yv= np.loadtxt("Y_test.csv", dtype=float, delimiter=',')
	return(xv.T,yv.T)    


#load weight of the DL in numpy format
def load_w_dae():
	data = np.load("wdae.npz")
	lst = data.files

	W = []
	for item in lst:
		W.append(data[item])
	
	return(W)    


def save_measure(cm,Fsc):
	
	np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')

	np.savetxt("fscores.csv", np.array(Fsc))
	return

# Beginning ...
def main():		
	xv,yv  = load_data_tst()
	W      = load_w_dae()
	cnf_dae, cnf_sft = ut.load_config()
	zv,a     = ut.dae_forward(xv,W[:-1],cnf_dae.encoder_act)
	z,a_soft = ut.forward_sft(a[-1],W[-1])
	cm,Fsc = ut.metricas(yv,a_soft[1])
	save_measure(cm,Fsc)
	print("Test finalizado correctamente.")
	print("Fscore promedio: ", Fsc[-1]*100)
	

if __name__ == '__main__':   
	 main()
