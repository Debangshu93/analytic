import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import scipy.linalg as scp
import os

def ts(dimension):
    d=dimension
    lmbda = 1.0
    answer = []
    theta_star = np.zeros((d,1)) 
    theta_star[0] = 1 
    X = theta_star
    X = X.T
    Y = np.array([np.dot(X[0,:].T,theta_star) + np.random.normal(0,1)])
    for n in range(1,5000):
        theta_hat = np.dot(np.linalg.inv(np.dot(X.T,X)+lmbda*np.identity(d)),np.dot(X.T,Y))
        beta = np.sqrt(2 * np.log((1+n/lmbda)/0.01)) + np.sqrt(lmbda)*np.linalg.norm(theta_star,2)
        std = np.sqrt(beta)*scp.sqrtm(np.linalg.inv(np.dot(X.T,X) + lmbda*np.identity(d)))
        var = np.linalg.matrix_power(std,2)
        theta_tilde = theta_hat.reshape(d,) + np.random.multivariate_normal(np.zeros((d,)),var).reshape(d,)
        x = (theta_tilde)/np.linalg.norm(theta_tilde,2)
        y = np.dot(x.reshape(1,d),theta_star) + np.random.normal(0,1)
        X = np.concatenate((X,x.reshape(1,d)), axis = 0)
        Y = np.concatenate((Y,y), axis = 0)
        G=np.dot(X.T,X)+lmbda*np.identity(d)
        eigenValues, eigenVectors = np.linalg.eig(G)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        answer.append((np.log(eigenValues[d-1]/np.sqrt(n))/np.log(d)))
        print(d, np.log(eigenValues[d-1]/np.sqrt(n))/np.log(d))
     
    return answer

def data_gen(sample, d):
    frame=[] 
    for _ in range(sample):
        frame.append(np.array(ts(d)))

    return frame

def plot(sample):
    f = plt.figure()
    for d in range(5,25,5):
        data = data_gen(sample, d)
        mean = np.mean(data, axis = 0)
        std = np.std(data,axis = 0)
        plt.plot(range(1001,5000), mean[1000:], label = "dimension = %s" %d, linewidth = 3)
        plt.fill_between(range(1001,5000), mean[1000:]-2*std[1000:], mean[1000:]+2*std[1000:], alpha = 0.40)
        plt.axhline(y=-1.5, color="black", linestyle="-", linewidth=5)
        plt.text(1000, -1.5, 'baseline=-1.5', fontsize=25, color = 'black')
        plt.ylabel(r"$\frac{\log{(n^{-\frac{1}{2}}\lambda_{\min}(V_n))}}{\log{d}}$", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xlabel(r"rounds ($n$) ", fontsize=40)
        plt.ylim([-1.5,0.8])
        plt.legend(loc = 'lower right', fontsize=30)
    plt.show()
    my_path = os.path.abspath(os.curdir)
    my_file = "gamma.pdf"
    print(os.path.join(my_path, my_file))
    f.savefig(os.path.join(my_path, my_file))
    
