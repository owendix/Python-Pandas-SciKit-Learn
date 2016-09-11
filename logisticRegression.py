# -*- coding: utf-8 -*-
"""Module for fitting a logistic regression 

Adapt for 3 dimensional plotting.

Created on Wed Dec 30 15:46:27 2015

@author: owendix
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
#for sparse data: lots of zeros
#from sklearn.preprocessing import MaxAbsScalar
#from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation


def mapFeature(x,y,degree=6,dtype=None):
    
    """Builds new features from existing features up to degree-th order
    
    Only works for 2 1-d arrays, much faster than multiMapFeature for this use
    
    Recommend degree <= 6
    """
    types=(int,float)
    #number of polynomial terms, not including x^0 or y^0
    n = int((degree*(degree+3))/2)
    if dtype in types:
        Z = np.zeros((x.shape[0],n),dtype=dtype)
    else:
        Z = np.zeros((x.shape[0],n))
    
    c = 0
    for i in range(1,degree+1):
        for j in range(i+1):
            Z[:,c] = (x**(i-j))*(y**j)
            c+=1
    
    return Z


def to_base_n(x,n=10,rvslist=True):
    
    s = []
    while x > 0:
        s.append(x%n)
        x = int(x/n)

    if rvslist:
        return s
    else:
        return ''.join(str(x) for x in s[::-1])

def incr_base_n(num,n=10,rvslist=True):
    
    if not rvslist:
        num = list(np.array(list(num[::-1]),dtype=int))
        
    i = 0
    i_prev = -1
    while i > i_prev:
        num[i] += 1
        if num[i] >= n:
            num[i] %= n
            i_prev = i
            i += 1
        else:
            i_prev = i
        
        if i >= len(num):
            num.append(1)
            if not rvslist:
                num = ''.join(str(x) for x in num[::-1])
                
            return num
            
    if not rvslist:
        num =  ''.join(str(x) for x in num[::-1])
    
    return num

def multiMapFeature(data, degree=6,dtype=None):
    
    """Maps the columns in data to the product of all combos of powers
    
    Outputs the product of all combinations of powers from 0th - degree-th
    Does not output the product of all 0th powers (always = 1)
    
    Slower than mapFeature for 2 1-d data. Outputs different order and
    up to a higher sum of powers than mapFeature
    
    """

    types=(int,float)
    
    if dtype in types:
        data = np.array(data,dtype=dtype)
    else:
        data = np.array(data)
 
    m, n = data.shape
    #m is number of examples
    #n is number of features (variables)

    shape = [degree+1]*n
    #create an n-dimensional square array of size degree+1
    if dtype in types:
        z = np.ones(shape,dtype=dtype)
    else:
        z= np.ones(shape)
    
    #fortran style column major (last index changing slowest)
    #e = np.ravel(z,order='F')[0]
    
    #try numpy meshgrid, or index_tricks.mgrid,??? couldn't figure it out
    if dtype in types:
        idx = np.zeros(n,dtype=dtype)
        Z = np.zeros((m,(degree+1)**n-1),dtype=dtype)
    else:
        idx = np.zeros(n)
        Z = np.zeros((m,(degree+1)**n-1))
    #z[tuple(idx)] #access value at index a
    #for all different dimensions, of which there are n=# of vbls/features
    #this accesses which index (0th, 1st, 2nd) in z[(0th,1st,2nd,...)]
    #need to iterate through all possible indices in array without an 
    #for loop for each index. Can use base-(degree+1) conversion from an
    #increasing number. This number cannot will be less than (degree+1)*n
    for i in range(m):
        #for every example, with n features
        if dtype in types:
            xs = np.array(data[i,:],dtype=int)
            idx = list(np.zeros(n,dtype=int))
        else:
            xs = np.array(data[i,:])
            idx = list(np.zeros(n))
        for j in range((degree+1)**n):
            #commented way is ~4X slower
            #idx=to_base_n(j,n=degree+1,rvslist=True)
            #need to pad with zeros up to length n
            #idx.extend([0]*(n-len(idx)))
            z[tuple(idx)] = np.product(xs**idx)
            idx = incr_base_n(idx,n=degree+1,rvslist=True)
       
        #copy dropping index (0,0,...,0), which
        #is the product of all features to the 0th power
        Z[i,:]=np.ravel(z,order='F')[1:]

    return Z


def plotDecisionBoundary(X,logreg,isMapped=False,ax=None,sf=0.15,hf=150):    
    

    x_min, x_max = X[:,0].min(), X[:,0].max()    
    s = sf*(x_max - x_min)    
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = x_min - s, x_max + s
    y_min, y_max = X[:, 1].min() - s, X[:, 1].max() + s
    
    h = (x_max-x_min)/hf    
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                         np.arange(y_min, y_max, h))
    if isMapped:
        zz = mapFeature(xx.ravel(),yy.ravel())
    else:
        zz = np.c_[xx.ravel(),yy.ravel()]            
    #concatenates along last axis (column)
    Z = logreg.predict(zz)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    if ax is None:
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
        plt.contour(xx,yy,Z,levels=[0.5],colors='k',linewidths=1.5)
    
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    else:
        ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
        ax.contour(xx,yy,Z,levels=[0.5],colors='k',linewidths=1.5)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    

    return ax


def plotLogitData(X,y,logreg,isMapped=False,title='Logistic Data',ax=None):
    
        
    n = X.shape[1]    
    
    if n < 4 or isMapped:
        #generate 1 figure
        fig = plt.figure()
        if ax is None:
            if n != 3:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111,projection='3d')
        else:
            plt.hold(True)
        
        ax.set_title(title)
        
        
        c = ['b','r','g','k','b','m','c','r','g','m']
        m = ['o','x','s','^','*','v','D','h','+','8']
        categs = len(set(y.ravel()))
        c = c[:categs]
        m = m[:categs]
        
        plt.hold(True)
        #plot decision boundary
        if logreg is not None:
            ax=plotDecisionBoundary(X,logreg,isMapped,ax=ax)
        
    if n == 1:
        ax.set_xlabel('x')
        ax.set_ylabel('y')    
               
        for i in range(categs):
            label = 'y = '+str(i)
            ax.scatter(X[y[:,0]==float(i), 0],y[y[:,0]==float(i)],
                         label=label,c=c[i],marker=m[i])
            
        #change color but not marker
        #ax3.scatter(X[:,0],X[:,1],y.ravel(),c=y.ravel(),cmap='bwr')         
    elif n == 2:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
       
        for i in range(categs):
            label = 'y = '+str(i)
            ax.scatter(X[y[:,0]==float(i), 0], X[y[:,0]==float(i), 1], 
                          label=label,c=c[i],marker=m[i])
       
    elif n == 3:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
            
        for i in range(categs):
            label = 'y = '+str(i)
            ax.scatter(X[y[:,0]==float(i), 0], 
                                 X[y[:,0]==float(i), 1],
                                 X[y[:,0]==float(i),2],label=label,
                                 c=c[i],marker=m[i])
            
    else:
        if isMapped:
            #assume only 2 base features polynomialized
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
       
            for i in range(2):
                label = 'y = '+str(i)
                ax.scatter(X[y[:,0]==float(i), 0], 
                                    X[y[:,0]==float(i), 1], 
                                    label=label,c=c[i],marker=m[i])
            
        else:
            print('Multidimensional X data')
            return None
    
    if n < 4 or isMapped:
        plt.legend(loc='best')
    
    return ax
    
def splitTraining(X, randomize=True):
    
    """Take input data and split off training data

    Assumes X is 2 dimensional, even if one dimension has length 1
    
    Suggest: split y off of each returned value after splitTraining called
    """    
    if randomize:
        #randomize along 0th axis (rows), modifies in place
        np.random.shuffle(X)
    

    c=0.7
    #could also split some for cross-validation - find optimized fit vals
    M = X.shape[0]
    m = int(c*M)
    #m_test = M - m
    return X[:m,:], X[m:,:]




def logisticRegression(X,y,C=None,useSKLearn=True,useScaler=True,
                       useMapFeature=False,isPlotted=False):

    """Add comments here and in logistic regression
    """

    
    #data must be in column form, comma separated entries
    
    #set to high number to reduce regularization, <~ 1e4
    useCV = C is None

        
    """
    General procedure:
    C is the relative weight given to the logistic fit, 
    relative to 1.0, the weight given to the regularization.
    --> high C means put less penalty on large value params
    --> should scale first with standard scalar, or minmax scalar or 
    max abs scalar
    
    #Should visualize decision boundary: 
    #be wary of over and underfitting
    
    """
    if useScaler:
        #to use the scaler for later test features or unscaling
        scaler = StandardScaler().fit(X)
        #scaler.mean_
        #scaler.scale_        
        X = scaler.transform(X)
    
    if useMapFeature:
        #only works for 2 features at a time
        X = mapFeature(X[:,0],X[:,1])
  
    #uses a random number generator to initialize
    #not exactly deterministic
    #try smaller tol if its too bad
    if useCV:
        clf_l2_LR = LogisticRegressionCV()
    else:
        clf_l2_LR = LogisticRegression(C=C, penalty='l2')
    #LogisticRegressionCV for cross-validation: picks best C
    
    clf_l2_LR.fit(X,y.ravel())
    
    #if useCV:
    #    print("C value:",clf_l2_LR.C_)
    
    
    #coefs = np.concatenate((clf_l2_LR.intercept_,
    #                        clf_l2_LR.coef_.ravel()),axis=0)
    
    #print('Coefficients')
    #print(coefs)
           
    if isPlotted:
        #plot decision boundary
        plotLogitData(X,y,clf_l2_LR,isMapped=useMapFeature)
        plt.show()
    
    return scaler, clf_l2_LR

        
def runLogReg(filename='SnowHazardData.txt',C=None,paramsArr=None,
              splitTrain=False,makeCrossVal=False, skiprows=0):
    
    """Use scaler to transform any other X's
    
        logreg.score(fit_transform(X),y)
    """
    
    data = np.loadtxt(filename, delimiter=',',skiprows=skiprows)
    #m data points, n variables to fit
    if splitTrain:
        Xs = splitTraining(data,makeCrossVal=makeCrossVal)
        if makeCrossVal:
            X, Xcv, Xtest = Xs
            ycv = Xcv[:,-1:]
            Xcv = Xcv[:,:-1]
        else:
            X, Xtest = Xs
        
        #split off y-data (best split first, randomized)
        y = X[:,-1:]
        X = X[:,:-1]
        ytest = Xtest[:,-1:]
        Xtest = Xtest[:,:-1]
    else:
        X = data[:,:-1] #2d array by slicing, column (m,n)
        y = data[:,-1:] #2d array by slicing, column (m,1):m data points
    
    #define my own measure for error (depends on difference between label
    #and prediction) - can use l2 or l1?
    #may need to use the sensitivity and specificity as a measure    
    
    scaler, logreg = logisticRegression(X,y,C=C,useScaler=True,
                                        isPlotted=False)
    
    #minimize error in Xcv to get best param
    #score attribute for cross validation
    if splitTrain:
        if makeCrossVal:
            Xs = [X, Xcv, Xtest]
            ys = [y, ycv, ytest]
        else:
            Xs = [X,Xtest]
            ys = [y,ytest]
    else:
        Xs = [X]
        ys = [y]
        
    return scaler, logreg, Xs, ys

    

    

                          
