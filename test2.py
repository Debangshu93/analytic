import numpy as np
d = 10
n = 20

X = np.random.rand(n,d)
# Let Y be a vector of ones of dimension n

    
inner_product = np.linalg.inv(np.dot(X.T,X) + np.identity(d))
matrix = np.dot(inner_product,np.dot(X.T,X)) - np.identity(d)
eig, eig_vec = np.linalg.eig(matrix)
