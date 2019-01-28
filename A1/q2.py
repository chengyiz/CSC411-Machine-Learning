# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    # solve (XtAX + lamdaI)w = XtAy
    # first find A with a(i)'s
    dis = l2(test_datum.T, x_train)
    noms = [-dis[0][i] / (2 * tau * tau) for i in range(x_train.shape[0])]

    denom = logsumexp(noms)
    
    # here we find the log of ai
    logais = [nom - denom for nom in noms]
    A = np.diag([np.exp(logai) for logai in logais])
        
    # build (XtAX + lambdaI)
    AX = np.dot(A, x_train)
    XtAX = np.dot(x_train.T, AX)
    lambdaI = lam * np.identity(d)
    left = XtAX + lambdaI
    
    # build XtAy
    right = np.dot(x_train.T, np.dot(A, y_train))
    w = np.linalg.solve(left, right)
    
    return np.dot(w, test_datum)
    ## TODO




def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    partition = N / k
    losses = np.zeros(200)
    for i in range(k):
        start = i * partition
        end = start + partition
        
        i_test = idx[start:end]
        
        # take one part as test
        x_test = x[i_test]
        y_test = y[i_test]
        
        # the rest as train
        x_train = np.delete(x, i_test, axis=0)
        y_train = np.delete(y, i_test)
        
        #print(i_test)
        losses += run_on_fold(x_test, y_test, x_train, y_train, taus)
        
    # find the avg
    losses = losses / float(k)
    
    return losses
    ## TODO


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)

    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(taus,losses)
    print("min loss = {}".format(losses.min()))
    plt.xlabel('taus')
    plt.ylabel('k-fold-loss')
    plt.show()