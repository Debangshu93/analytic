from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.linalg as scp
import os
 


def ts(dimension):
    d=dimension
    lmbda = 1.0
    answer = []
    theta_star = np.ones((d,1)) 
    
    X = np.zeros((d,1))
    X[0] = 1
    X = X.T
    Y = np.array([np.dot(X[0,:].T,theta_star) + np.random.normal(0,1)])
    for n in range(1,5000):
        print(d, n) 
        theta_hat = np.dot(np.linalg.inv(np.dot(X.T,X)+lmbda*np.identity(d)),np.dot(X.T,Y))
        beta = np.sqrt(2 * np.log((1+n/lmbda)/0.01)) + np.sqrt(lmbda)
        std = np.sqrt(beta)*scp.sqrtm(np.linalg.inv(np.dot(X.T,X) + lmbda*np.identity(d)))
        var = np.linalg.matrix_power(std,2)
        theta_tilde = theta_hat.reshape(d,) + np.random.multivariate_normal(np.zeros((d,)),var).reshape(d,)
        
        x_opt = cp.Variable(d)
       
        objective = cp.Maximize(x_opt.T @ theta_tilde)
        constraint = [cp.norm(x_opt, 10)<=1]
        cp.Problem(objective, constraint).solve()
        x = x_opt.value
        y = np.dot(x.reshape(1,d),theta_star) + np.random.normal(0,1)
        
        
        X = np.concatenate((X,x.reshape(1,d)), axis = 0)
        Y = np.concatenate((Y,y), axis = 0)
        G=np.dot(X.T,X)+lmbda*np.identity(d)
        eigenValues, eigenVectors = np.linalg.eig(G)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        print(np.log(eigenValues[d-1])/np.log(n))
        
        answer.append(np.log(eigenValues[d-1])/np.log(n))
         

    
    return answer
    

def data_gen(sample, d):
    frame=[] 
    for _ in range(sample):
        frame.append(np.array(ts(d)))

    return frame


def plot(sample):
    f = plt.figure()
    for d in [5, 20]:
        data = data_gen(sample, d)
        mean = np.mean(data, axis = 0)
        std = np.std(data,axis = 0)
        plt.plot(mean, label = "dimension = %s" %d, linewidth = 3)
        plt.fill_between(range(1,5000), mean-2*std, mean+2*std, alpha = 0.20)
        plt.axhline(y=0.5, color="black", linestyle=":", linewidth=4)
        plt.ylabel(r"$\frac{\log{(\lambda_{\min}(V_n))}}{\log{n}}$", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xlabel(r"rounds ($n$) ", fontsize=40)
        plt.legend(loc = 'best', fontsize = 30)
    plt.show()
    my_path = os.path.abspath(os.curdir)
    my_file = "convex.pdf"
    print(os.path.join(my_path, my_file))
    f.savefig(os.path.join(my_path, my_file))
    