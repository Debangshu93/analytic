import numpy as np
import matplotlib.pyplot as plt

#Let n be the number of data
# generate n such random vactors of dimension d
# Call it X of dimension n \times d
d = 2


for n in range(8,1000):
	n_1 = n**(2/3)
	n_2 = n-n_1
	vector_1 = np.array([1.0,0], dtype=np.float32)
	matrix_1 = np.tile(vector_1, (int(n_1),1))
	vector_2 = np.array([np.sqrt(1 - 1/n_2), np.sqrt(1/n_2)], dtype = np.float32)
	matrix_2 = np.tile(vector_2,(int(n_2),1))
	X = np.concatenate((matrix_1,matrix_2), axis=0)

# Let Y be a vector of ones of dimension n

	U,S,V = np.linalg.svd(X)
	answer = np.linalg.norm(X[:,1],1)/np.linalg.norm(X[:,1],2)**2
    
	plt.scatter(n,answer)
    

plt.show()
