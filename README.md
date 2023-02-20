# Experiments for the paper "Exploration in Linear Bandits with rich action sets and its implications for inference" to appear in 26th AISTATS Conference 2023.
Images for the results in the paper "Exploration in Linear Bandits in Rich Action Sets and its implications for inference", to appear in the 26th AISTATS Conference.


Run driver.py as python driver.py from terminal to get all results.

Run lambda_min_sphere.py to generate figure 2 (Minimum eigen value of Sphere). Image saved as d3, d5, d10, d20 pdf files.

Run plot_for_n0.py to generate left image of figure 3 (How n_0 varies). Image saved as n_0.pdf.

Run plot_for_gamma.py to generate right image of figure 3 (How \gamma varies). Image saved as gamma.pdf.

Run lambda_min_convex.py to generate figure 5 (Minimum eigen-value of l_10 ball). Image saved as convex.pdf.

To run each program separately run it from a python interpreter. Import the respective python module and run the plot() method.
Example to run say plot_for_n0.py, go to a python interpreter.

Run :

import plot_for_n0 as TS

TS.plot(20)

plot() def takes input as an integer constant which represt the number of trials you want to experiment over.

lambda_min_convex.py requires cvxpy python package.
