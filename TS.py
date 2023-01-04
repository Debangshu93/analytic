import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scp

lmbda = 1.0

theta_star = np.array([1.0,0.0])

X = np.array([[np.sqrt(1-1/1), np.sqrt(1/1)]])
#G = np.dot(X[0,:].reshape(2,1),theta_star.reshape(1,2)) + np.dot(theta_star.reshape(2,1),X[0,:].reshape(1,2))

Y = np.array([np.dot(X[0,:].T,theta_star) +10*np.sin(np.dot(X[0,:].T,theta_star)) + np.random.normal(0,1)])
Y_star = np.array([np.dot(theta_star.T,theta_star) + 10*np.sin(np.dot(theta_star.T,theta_star)) + np.random.normal(0,1)])

for n in range(1,1000):
    theta_hat = np.dot(np.linalg.inv(np.dot(X.T,X)+lmbda*np.identity(2)),np.dot(X.T,Y))
    beta = np.sqrt(2 * np.log((1+n/lmbda)/0.01)) + np.sqrt(lmbda)*np.linalg.norm(theta_star,2)
    std = np.sqrt(beta)*scp.sqrtm(np.linalg.inv(np.dot(X.T,X) + lmbda*np.identity(2)))
    theta_tilde = theta_hat + np.random.multivariate_normal([0,0],std)
    x = (theta_tilde)/np.linalg.norm(theta_tilde,2)
    
    y = np.dot(x.reshape(1,2),theta_star) + 10*np.sin(np.dot(x.reshape(1,2),theta_star)) + np.random.normal(0,1)
    y_star = np.array([np.dot(theta_star.T,theta_star)+ 10*np.sin(np.dot(theta_star.T,theta_star)) + np.random.normal(0,1)])
    
    X = np.concatenate((X,x.reshape(1,2)), axis = 0)
    Y = np.concatenate((Y,y), axis = 0)
    Y_star = np.concatenate((Y_star,y_star), axis = 0)
    
    #G = G + np.dot(x.reshape(2,1),theta_star.reshape(1,2)) + np.dot(theta_star.reshape(2,1),x.reshape(1,2))
    #eigen_values_1, eigen_vectors_1 = np.linalg.eig(G)
    
    answer1 = np.linalg.norm(theta_tilde,2)
    plt.scatter(n,answer1)	
    
plt.show()
