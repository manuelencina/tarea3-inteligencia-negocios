import numpy as np
import utility as ut


#load data for testing
def load_data_tst():
	xv= np.loadtxt("Xv.csv", dtype=float, delimiter=',')
	yv= np.loadtxt("Yv.csv", dtype=int, delimiter=',')
	return(xv,yv)    


#load weight of the DL in numpy format
def load_w_dl():
	data = np.load("wAEs.npz")
	lst = data.files

	W = []
	for item in lst:
		W.append(data[item])

	data2 = np.load("wSoftmax.npz")
	lst2 = data2.files

	for item2 in lst2:
		W.append(data2[item2])
	
	return(W)    

# Feed-forward of the DL
def forward_dl(x,W):
	par_sae, par_sft = ut.load_config()
	w_size = len(W)
	for i in range(0, w_size):
		z = np.matmul(W[i], x)
		
		if i == w_size -1:
			x = np.copy(z)
			break
		
		x = ut.act_function( par_sae["activation_function"], z)

	return(x)


def save_measure(cm,Fsc):
	
	np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')

	np.savetxt("fscores.csv", np.array(Fsc))
	return

# Beginning ...
def main():		
	xv,yv  = load_data_tst()
	W      = load_w_dl()
	zv     = forward_dl(xv,W)
	cm,Fsc = ut.metricas(yv,zv)
	save_measure(cm,Fsc)
	print("Test finalizado correctamente.")
	print("Fscore promedio: ", Fsc[-1]*100)
	

if __name__ == '__main__':   
	 main()
