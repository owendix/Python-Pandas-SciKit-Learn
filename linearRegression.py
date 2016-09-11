# -*- coding: utf-8 -*-
"""Module for fitting a linear regression 

Adapt for 3 dimensional plotting.

Created on Wed Dec 30 15:46:27 2015

@author: owendix
"""
import numbers
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


def solveLinRegNormal(X,y,reg=False,C=None):
    
    """Compute coefficients of multivar-linear regression with normal eq
    
    
    The normal equation is a faster method than computing the coefficients
    of linear regression than gradient descent and does not require scaling
    Note, matrix inverse may be numerically unstable. The Moore-Penrose
    pseudo inverse of a matrix is actually better if noninvertible, in 
    this case
    
    If the matrix is not invertible, there are 2 common causes:
    1) Redundant, linearly dependent, features: e.g. size in feet and meters
        --> Delete one
    2) Too many features, m <= n: fitting more variables than data points
        --> Delete some features (better) or use regularization - sometimes
        works but is not a good idea with too little data, get more data!
    
    Arguments:
    X -- multivariable X data as numpy array
    y -- output data to match
    reg -- regularization, default False
    C -- coefficient regularization from sklearn (1/lambda, Andrew Ng)
    
    Return:
    coefficients, (n+1 x 1) numpy array
    
    """    

    if not reg:  
        #X is (m,n+1)
        #y is (m,1)
        
        #matrix pseudoinverse (uses svd)
        #((n+1,m)(m,n+1))(n+1,m)(m,1)
        #(n+1,n+1)(n+1,1)
        coefs = (np.linalg.pinv(X.T.dot(X))).dot(X.T.dot(y)) #(n+1,1)
    else:
        if C is not None and isinstance(C,numbers.Real) and C > 0.:
            lam = np.identity(X.shape[1])/C
        else:
            print('C must be a positive number')
            sys.exit()
            
        coefs = (np.linalg.pinv(X.T.dot(X) + lam)).dot(X.T.dot(y))
    
    return np.asarray(coefs) #(n+1,1)

def plotLinPrediction(X,prediction,ax=None):

    #do not pass prepended array of 1s
    #number of features
    m,n = X.shape
    print('m,n=',m,n)
    print('prediction shape:',prediction.shape)
    if ax is None:
        if n <= 1:
            ax = plt.subplot(111)
        elif n == 2:
            ax = plt.subplot(111,projection='3d')
    else:
        plt.hold(True)
    
    #marker, color string
    if m < 2:
        s = 'ro'
    else:
        s = 'r'    
    
    if n <= 1:
        ax.plot(X.ravel(),prediction.ravel(),s)
    elif n == 2:
        #note, in three dimensions (x1,x2,y) the solution 
        #takes the form: y = b + x1*w1 + x2*w2
        #if you plug in a specific x1, x2, you get a specific y
        #Rearranging:
        #w1*x1 + w2*x2 + (-1)*y + b = 0
        # this is the equation of a plane with normal vector:
        #n = [w1,w2,-1]
        #not sure how to pick out the line here
        ax.plot(X[:,0],X[:,1],prediction.ravel(),s)
    else:
        print('Multidimensional X data')
    
    if n < 3:
        return ax
    else:
        return None
    

def plotLinData(X,y,title='Linear Regression Data'):
    
    n = X.shape[1]
    if n < 3:
        fig = plt.figure()
    if n == 1:
        #generate 1 figure
        ax = fig.add_subplot(111)
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')    
        ax.set_title(title)
        ax.scatter(X,y)
    elif n == 2:
        #preferred method to generate a 3d figure
        ax = fig.add_subplot(111,projection='3d')
    
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title(title)
        ax.scatter(X[:,0],X[:,1],y.ravel())

    else:
        print('Multidimensional X data')
        
    if n < 3:
        return ax
    else:
        return None

def linearRegression(filename='ex1data1.txt',useSKLearn=True):
    
    #data must be in column form, comma separated entries
 
    if useSKLearn:
        print('Using scikit-tools learn to fit data')
    
    
    data = np.loadtxt(filename, delimiter=',')
    #m data points, n variables to fit
    X = data[:,:-1] #2d array by slicing, column (m,n)
    y = data[:,-1:] #2d array by slicing, column (m,1):m data points

    if not useSKLearn:
        #need X to have 1s in first column (y-intercept)
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        coefs = solveLinRegNormal(X,y)
        #plot fit line, X has prepended 1s not needed in indep var
        print('Coefficients:',coefs)
        
        axis = plotLinData(X[:,1:],y,title=filename)
        axis = plotLinPrediction(X[:,1:],X.dot(coefs),ax=axis)
        
        #some random prediction:
        print('Some Prediction: [newX],[y]')
        newX = np.ones((1,X.shape[1]))
        for i in range(1,newX.shape[1]):
            x_min, x_max = X[:,i].min(), X[:,i].max()
            s = 0.2*(x_max - x_min)
            x_min = int(x_min - s)
            x_max = int(x_max + s)
            newX[0,i] = np.random.randint(x_min,x_max)
        
        #-1 infer from data
        
        prediction = newX.dot(coefs)
        print(newX[:,1:],prediction)

        plotLinPrediction(newX[:,1:],prediction,ax=axis)
        if X.shape[1] < 4:
            plt.show()
        
    else:
        clf = linear_model.LinearRegression()
        #gives optional parameters:
        #like to normalize before fitting: no need
        #clf.fit(X,y,normalize=True)
        #set y intercept = 0
        #clf.fit(X,y,fit_intercept=False)
        clf.fit(X,y)
        
        coefs = np.concatenate((clf.intercept_,
                                clf.coef_.ravel()),axis=0)
        print('Coefficients:',coefs)

        axis = plotLinData(X,y,title=filename)
        axis = plotLinPrediction(X,clf.predict(X),ax=axis)

    
        #can predict for X = [new #s]
        print('Some Prediction: [newX],[y]')
        newX = np.ones((1,X.shape[1]))
        for i in range(newX.shape[1]):
            x_min, x_max = X[:,i].min(), X[:,i].max()
            s = 0.2*(x_max - x_min)
            x_min = int(x_min - s)
            x_max = int(x_max + s)
            newX[0,i] = np.random.randint(x_min,x_max)
      
        print(newX,clf.predict(newX))

        plotLinPrediction(newX,clf.predict(newX),ax=axis)
        if X.shape[1] < 4:
            plt.show()

