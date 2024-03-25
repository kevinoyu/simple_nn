# simple_nn

Simple autogradient library and nn model from scratch for educational purposes, following similar-ish interface to pytorch.

src/gradient: code for a basic autogradient library built on top of np arrays. Basic support for gradient backpropogation for matrix operations such as addition, matrix multiplication, and relu's - the minimum requirements for a basic perceptron NN with nonlinearity. 

src/nn: code for a basic neural network (linear layers, relus) + training utilities including a simple MSE loss function gradient descent optimizer.
