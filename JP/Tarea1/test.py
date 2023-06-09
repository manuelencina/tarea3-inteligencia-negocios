import numpy as np

A = [[0.1, 0.2, 0.3, 0.99, 0.98], [0.1, 0.2, 0.3,0, 0.98], [0.1, 0.2, 0.99, 0.1, 0.98]]

A = np.array(A)

for sample in A:
    max_value_i = np.argmax(sample)
    sample[max_value_i] = 1





print(A.astype(int))