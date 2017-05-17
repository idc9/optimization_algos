
This repo contains basic implementations of a few optimization algorithms. The primary purpose of this repo is for me to learn and have bare bones implemtations sitting around for future use.

The important folders are:

- **models**: This folder contains a few simple optimiation models (e.g. least squares, logistic regression). Each model is an object and has functions such as F or grad_F (the function value and the gradient) that return data used by various optimization algorithms. 

- **opt_algos**: Various optimiation algorithms (gradient descent, SGD, etc) are implemented in python. Each optimization function takes a model object (see above) as an argument plus various optimization hyperparameters (e.g. learning rate, max iterations, etc). 


Notation: since I am a statistician I am using beta for the function variables, and X/y for the data for the optimization problem. Capital **F** is the objective function (usually a negative log likelihood). Lower case *f* is the likelihood of an indivudual data point for [ERM](https://cs.stanford.edu/~quocle/TeoLeSmoVis07.pdf) problems (e.g. least squares). **eta** is the learning rate. **L_F** and **mu_F** are the Lipshitiz and strong convexity constants for the objectie fuction.

---

# Refernces

Most of the code is based on the lecture notes from STOR 892: Convex Optimization. Some additional useful references include

- [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/index.html) by Sebastian Ruder