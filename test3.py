import numpy as np
import matplotlib.pyplot as plt

#Let n be the number of data
# generate n such random vactors of dimension d
# Call it X of dimension n \times d
d = 50


for n in range(1,3000):

    diag = np.random.multinomial(n,[1/d]*d)
    X = np.diag(diag)
# Let Y be a vector of ones of dimension n

    Y = np.ones((d,1))
    inner_product = X + np.identity(d)
    eigen_values = np.linalg.eigvalsh(inner_product)
    answer = np.amin(eigen_values)
    plt.scatter(n,answer)
    plt.scatter(n,n/d)

plt.show()
