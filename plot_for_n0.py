from turtle import color
import numpy as np
import matplotlib.pyplot as plt
#import cvxpy as cp
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
    for n in range(1,1200):
        print(d, n) 
        theta_hat = np.dot(np.linalg.inv(np.dot(X.T,X)+lmbda*np.identity(d)),np.dot(X.T,Y))
        beta = np.sqrt(2 * np.log((1+n/lmbda)/0.01)) + np.sqrt(lmbda)
        std = np.sqrt(beta)*scp.sqrtm(np.linalg.inv(np.dot(X.T,X) + lmbda*np.identity(d)))
        var = np.linalg.matrix_power(std,2)
        theta_tilde = theta_hat.reshape(d,) + np.random.multivariate_normal(np.zeros((d,)),var).reshape(d,)
        
        x = theta_tilde/np.linalg.norm(theta_tilde,2)
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
    c= ['blue', 'orange', 'green', 'red']
    f = plt.figure()
    j=0
    #d = 10
    for d in [5,10, 15, 20]:
        data = data_gen(sample, d)
        mean = np.mean(data, axis = 0)
        for i in range(len(mean)):
            if mean[i] > 0.5:
                if(d==5):
                    plt.axvline(x=i, linestyle=":", color = c[j], linewidth = 4)
                    plt.text(i, 0.2, r"$n_0=$ %s" %i, fontsize = 20,  color = c[j])
                    j=j+1
                    break
                elif (d==10):
                    plt.axvline(x=i, linestyle=":", color = c[j], linewidth = 4)
                    plt.text(i, 0.1, r"$n_0=$ %s" %i, fontsize = 20,  color = c[j])
                    j=j+1
                    break
                elif(d==15):
                    plt.axvline(x=i, linestyle=":", color = c[j], linewidth = 4)
                    plt.text(i, 0.2, r"$n_0=$ %s" %i, fontsize = 20,  color = c[j])
                    j=j+1
                    break
                else:
                    plt.axvline(x=i, linestyle=":", color = c[j], linewidth = 4)
                    plt.text(i, 0.4, r"$n_0=$ %s" %i, fontsize = 20,  color = c[j])
                    j=j+1
                    break
        std = np.std(data,axis = 0)
        plt.plot(mean, label = "dimension = %s" %d, linewidth = 3)
        plt.fill_between(range(1,1200), mean-2*std, mean+2*std, alpha = 0.20)
        plt.axhline(y=0.5, color="black", linestyle=":", linewidth=4)
        plt.ylabel(r"$\frac{\log{(\lambda_{\min}(V_n))}}{\log{n}}$", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xlabel(r"rounds ($n$) ", fontsize=40)
        plt.legend(loc = 'best', fontsize = 30)
    plt.show()
    my_path = os.path.abspath(os.curdir)
    my_file = "n_0.pdf"
    print(os.path.join(my_path, my_file))
    f.savefig(os.path.join(my_path, my_file))
    