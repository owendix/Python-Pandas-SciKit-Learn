# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:37:50 2016

@author: owendix
"""

import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete


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
    

def splitTraining(X, makeCV=False, randomize=True):
    
    """Take input data and split off training data

    Assumes X is 2 dimensional, even if one dimension has length 1
    
    Suggest: split y off of each returned value after splitTraining called
    """    
    if randomize:
        #randomize along 0th axis (rows), modifies in place
        np.random.shuffle(X)
    
    if makeCV:
        c1 = 0.6
        c2 = 0.8
        M = X.shape[0]
        m1 = int((c1*M))
        m2 = int(c2*M)
        
        return X[:m1,:], X[m1:m2,:], X[m2:,:]
    else:
        c=0.7
        #could also split some for cross-validation - find optimized fit vals
        M = X.shape[0]
        m = int(c*M)
        #m_test = M - m
        return X[:m,:], X[m:,:]

def getError(predictions,actuals):
    
    m = len(predictions)
    return np.sum(np.subtract(predictions,actuals)**2)/(2*m)


def getErrorProb(suppt,predictions,actuals,return_suppt=True):
    
    """Return the error joint, discrete sample probability estimates
    
    Arguments:
    suppt: list of support values
    
    
    #Cannot use strings or tuples as support values: use integers
    Return:
    new_suppt, err_prob = getError(suppt,preds,actls)

    #
    p(diff,pred) = err_prob.pmf(new_suppt)
    #want a column of rows representing each diff value
    p(diff=0,1,2,...) = np.sum(err_prob.pmf(new_suppt),axis=1).reshape(-1,1)
    #want a row representing p for each predicted value
    p(pred=0,1,2,...) = np.sum(err_prob.pmf(new_suppt),axis=0)
    p(diff=0,1,2...|pred=0,1,2,) = p(diff,pred)/p(pred=0,1,2,...)
       = err_prob.pmf(new_suppt)/np.sum(err_prob.pmf(new_suppt),axis=0)
    #will get some nan's if dividing by 0 (no chance of getting pred=x)
    #if you sum over all diff values (rows) for a fixed pred value (col)
    # you must get 1.0 (and you do!)
       
    #I should turn this into a class
    #This way I could store the methods to find the useful probabilities
    #In the meanwhile, I'd like some experience with pandas relational algebra

    """
    #vals = pd.DataFrame([predictions,actuals],columns=['predict','actual'])
    #not really useful    
    
    n = len(suppt)
    min_ndata = min([len(predictions),len(actuals)])
    if n <= 10 and min_ndata > 0:
        #choose these specific values: use 1-dig integers as symbols
        new_suppt = np.arange(n)
        xs = np.array([i*10 + new_suppt for i in range(n)])
        
        ps = np.zeros(xs.shape)
        for p,a in zip(predictions,actuals):
            #requires they are numerical values
            d = int(abs(p - a))
            #return index, new support value
            v = suppt.index(p)
            ps[d,v] += 1.
        

        ps /= min_ndata
        distrib = rv_discrete(values=(xs,ps))
        if return_suppt:
            return xs, distrib
        else:
            return distrib
    else:
        return None


def loadAndTrainHazard(hazard='Snow',suffix='1',mLim=0,multiMapFeat=False,
                       mapFeat=False,polyd=4,rand=False,split=False,
                       makeCV=False,returnData=False,method='svm',
                       kernel='linear',C=None):
    
    filename = hazard+'HazardData'+suffix+'.txt'
    
    data = np.loadtxt(filename,delimiter=',',skiprows=0)
    if multiMapFeat:
        #don't take the y-data
        Z = multiMapFeature(data[:,:-1],degree=polyd,dtype=int)
        data = np.concatenate((Z,data[:,-1:]),axis=1)
    elif mapFeat:
        #want to map the new snow and temperature data, cols 1 and 2
        Z = mapFeature(data[:,1],data[:,2],degree=polyd,dtype=int)
        data = np.concatenate((data[:,0:1],Z,data[:,3:]),axis=1)
    if rand:
        #randomizes in place row locations, not entry locations within rows
        np.random.shuffle(data)
    if split:
        dataVec = splitTraining(data,makeCV=makeCV,randomize=False)
        if makeCV:
            data, dataCV, dataTest = dataVec
        else:
            data, dataTest = dataVec
    
    X,y = data[:,:-1], data[:,-1]
    if mLim > 0:
        mLim = min(mLim,len(y))
        X,y = X[:mLim,:], y[:mLim]
    scaler = StandardScaler().fit(X)
    if 'svm' in method:
        fit = svm.SVC(kernel=kernel,C=1.).fit(scaler.transform(X),y)
    else:
        if C is None:
            fit = LogisticRegressionCV().fit(X,y)
        else:
            fit = LogisticRegression(C=C,penalty='l2').fit(X,y)
    
    if returnData:
        if not split:
            return scaler, fit, X, y
        else:
            Xtest = dataTest[:,:-1]
            ytest = dataTest[:,-1]
            if makeCV:
                Xcv = dataCV[:,:-1]
                ycv = dataCV[:,-1]
                
                return scaler,fit,X,Xcv,Xtest,y,ycv,ytest
            else:
                return scaler,fit,X,Xtest,y,ytest
    else:
        return scaler, fit


def trainAndPredictHazards(suffix='1'):
    
    hazardType = ['Snow','Road']    
    nH = len(hazardType)
    
    #1st row is today's data, since it predicts tomorrow
    #which we don't have hazard target values, it is not used to train
    predsIn = np.array([[2.,0.,24.,2.,0.],
                      [3.,0.,47.,2.,0.],
                      [4.,0.,48.,2.,0.],
                      [5.,0.,39.,2.,0.],
                      [6.,0.,43.,2.,0.]
                      ])
    nP = predsIn.shape[0]
    predsOut = np.zeros((nP,nH))
    
    for d in range(nP):
        if d != 0:
            #stick in for next prediction
            predsIn[d,-nH:] = predsOut[d-1,:]
        X_test = predsIn[d,:].reshape(1,-1)
        for j, hzt in enumerate(hazardType):
            scaler, svmFit = loadAndTrainHazard(hazard=hzt,suffix=suffix,
                                                rand=False,returnData=False
                                                )

            predsOut[d,j] = svmFit.predict(scaler.transform(X_test))
    
    return predsIn, predsOut
            

def plotLearningCurves(xRange,Jerr=True,x_means='Day',fb='Snow',C=1.,
                       method='svm',kernel='linear',degree=3,makeCV=False,
                       shift_ns=True,pplot=False,ntrials=10,
                       rand=True,insuffix=1,outsuffix=1):

    """
    Arguments:
    x_means=['Day','m','C','deg']
    method: ['svm','logistic']
    if 'svm':
    kernel: ['linear','rbf','poly,'sigmoid']
    """
    xLen = len(xRange)
    scores_train = np.zeros((xLen,ntrials))
    scores_test = np.zeros((xLen,ntrials))
    err_prob_train = np.zeros((xLen,ntrials)).tolist()
    err_prob_test = np.zeros((xLen,ntrials)).tolist()
    
    if 'Snow' not in fb:
        fb = 'Road'
        suppt = [0,1,2,3,4]
    else:
        fb = 'Snow'
        suppt = [0,1,2,3,4,5]
    
    fb = fb + 'HazardData'
    if shift_ns:
        fb = fb + 'ShiftNS'
    
    if 'Day' not in x_means:
        if insuffix is None:
            f = fb + '.txt'
        else:
            f = fb + str(insuffix) + '.txt'
        all_data = np.loadtxt(f,delimiter=',',skiprows=0)
    
    rtnsuppt=True
    if Jerr:
        best_score=1e5
        worst_score=0.
    else:
        best_score=0.
        worst_score=1.
    
    for n,x in enumerate(xRange):
        print('x = ',x)
        if 'Day' in x_means:
            f = fb + str(x) + '.txt'
            data = np.loadtxt(f,delimiter=',',skiprows=0)
        elif 'm' in x_means:
            data = all_data[:x,:]
        elif 'C' in x_means:
            data = all_data
            C = x
        elif 'deg' in x_means:
            degree=x
            data = all_data
            if 'svm' not in method:
                #using logistic regression, degree used for multiMapFeature
                scaler = StandardScaler().fit(data[:,:-1])
                #scaler.mean_
                #scaler.scale_        
                data[:,:-1] = scaler.transform(data[:,:-1])
                Z = multiMapFeature(data[:,:-1],degree=degree,dtype=int)
                #I may need to scale ahead of time
                data = np.concatenate((Z,data[:,-1:]),axis=1)
                
        else:
            data = all_data
        
        if 'deg' not in x_means or 'svm' in method:
            scaler = StandardScaler().fit(data[:,:-1])
            #scaler.mean_
            #scaler.scale_        
            data[:,:-1] = scaler.transform(data[:,:-1])          
            
    
        for i in range(ntrials):
            if makeCV:
                X_train,X_CV,X_test = splitTraining(data,makeCV=False,
                                           randomize=rand
                                           )
                y_train, y_CV, y_test = X_train[:,-1], X_CV[:,-1],X_test[:,-1]
                X_train, X_CV,X_test = X_train[:,:-1],X_CV[:,:-1],X_test[:,:-1]                
            else:
                X_train,X_test = splitTraining(data,makeCV=False,
                                           randomize=rand
                                           )
                y_train, y_test = X_train[:,-1], X_test[:,-1]
                X_train, X_test = X_train[:,:-1], X_test[:,:-1]
                

            if 'svm' in method:
                if C is None and 'C' not in x_means:
                    C = 1.
                clf = svm.SVC(kernel=kernel,degree=degree,C=C
                            ).fit(X_train,y_train)
            else:
                if C is not None:
                    clf = LogisticRegression(C=C).fit(X_train,y_train)
                else:   
                    clf = LogisticRegressionCV().fit(X_train,y_train)
            if Jerr:
                scores_train[n,i] = getError(clf.predict(X_train),y_train)
                scores_test[n,i] = getError(clf.predict(X_test),y_test)
                if scores_test[n,i] < best_score:
                    best_predict = clf.predict(X_test)
                    best_y = y_test.copy()
                    best_score = scores_test[n,i]
                    best_x = x
                elif scores_test[n,i] > worst_score:
                    worst_predict = clf.predict(X_test)
                    worst_y = y_test.copy()
                    worst_score = scores_test[n,i]
                    worst_x = x

            else:
                scores_train[n,i] = clf.score(X_train,y_train)
                scores_test[n,i] = clf.score(X_test,y_test)
                if scores_test[n,i] > best_score:
                    best_predict = clf.predict(X_test)
                    best_y = y_test.copy()
                    best_score = scores_test[n,i]
                    best_x = x
                elif scores_test[n,i] < worst_score:
                    worst_predict = clf.predict(X_test)
                    worst_y = y_test.copy()
                    worst_score = scores_test[n,i]
                    worst_x = x
            
            if rtnsuppt:
                rtnsuppt = False
                err_prob_suppt, err_prob_train[n][i] = getErrorProb(suppt,
                                                clf.predict(X_train), y_train
                                                )
            else:
                err_prob_train[n][i] = getErrorProb(suppt,
                                                clf.predict(X_train), y_train,
                                                return_suppt=False
                                                )
            err_prob_test[n][i] = getErrorProb(suppt,clf.predict(X_test),
                                                y_test,return_suppt=False
                                                )


    #data stats
    mdn_train = np.median(scores_train,axis=1)
    mdn_test = np.median(scores_test,axis=1)
    factor = 1.4826
    mad_train = factor*np.median(np.abs(scores_train - mdn_train.reshape(-1,1)
                                        ),axis=1
                                )
    mad_test = factor*np.median(np.abs(scores_test - mdn_test.reshape(-1,1)
                                        ),axis=1
                                )
 
    #plot data
    fig, [ax1,ax2,ax3] = plt.subplots(nrows=3,ncols=1)
    

    
    ax1.errorbar(xRange,mdn_train,yerr=mad_train,marker='o',color='b',
                 label='Training'
                 )
    ax1.errorbar(xRange,mdn_test,yerr=mad_test,marker='o',color='r',
                 label='Test'
                 )
    
    ax1.set_xlabel(x_means)
    if not Jerr:
        ax1.set_ylabel('Median Accuracy')
    else:
        ax1.set_ylabel('Median Error')
    if 'svm' in method:
        ax1.set_title(fb+' SVM-' + kernel+' Predictions- ' + str(ntrials)+
                        ' Trials'+str(outsuffix)
                        )
    else:
        ax1.set_title(fb+' LogisticRegression Prediction- ' + str(ntrials)+
                        ' Trials'+str(outsuffix)
                        )
    if 'C' in x_means:
        ax1.set_xscale("log", nonposx='clip')
        #xmin,xmax = min(xRange), max(xRange)
        #ax1.set_xlim([xmin,xmax])
    else:
        xmin,xmax = min(xRange), max(xRange)
        xdiff = xmax - xmin
        xmin, xmax = xmin - 0.1*xdiff, xmax + 0.1*xdiff
        ax1.set_xlim([xmin,xmax])
    if not Jerr:
        ax1.set_ylim([0.0,1.1])
    else:
        ymax=1.2*(mdn_test.max()+mad_test.max())
        ax1.set_ylim([0.0,ymax])
    ax1.grid(True,axis='y')
    ax1.legend(fontsize=9,loc='best')
    
    #axis 2: best predictions
    ax2.plot(best_predict,marker='o',color='b',
             label='Predict, Score='+'{:.2f}'.format(best_score)
             )
    ax2.plot(best_y,marker='o',color='r',label='Actual')
    
    if 'Road' in fb:
        ax2.set_ylim([-0.5,8])
    else:
        ax2.set_ylim([-0.5,9])
    ax2.set_xlabel('#')
    ax2.set_ylabel(fb)
    if 'Day' in x_means:
        ax2.set_title('Best Test Prediction, '+str(best_x)+' Days')
    else:
        ax2.set_title('Best Test Prediction, '+str(best_x))
    ax2.legend(fontsize=9,loc='best')
    ax2.grid(True,axis='y')
    
    #axis 3: worst predictions
    ax3.plot(worst_predict,marker='o',color='b',
             label='Predict, Score='+'{:.2f}'.format(worst_score)
             )
    ax3.plot(worst_y,marker='o',color='r',label='Actual')
    
    if 'Road' in fb:
        ax3.set_ylim([-0.5,10])
    else:
        ax3.set_ylim([-0.5,10])
    ax3.set_xlabel('#')
    ax3.set_ylabel(fb)
    if 'Day' in x_means:
        ax3.set_title('Worst Test Prediction, '+str(worst_x)+' Days')
    else:
        ax3.set_title('Worst Test Prediction, '+str(worst_x))
    ax3.legend(fontsize=9,loc='best')
    ax3.grid(True,axis='y')
    
    plt.tight_layout(h_pad=0.0)
    if not pplot:
        plt.show()
    else:
        outsuffix = x_means + str(outsuffix)
        if 'svm' in method:
            plt.savefig(fb+'SVM'+kernel+str(outsuffix)+'.png')
        else:
            plt.savefig(fb+'LogReg'+str(outsuffix)+'.png')
     
    plt.close(fig)
      
    return err_prob_suppt, err_prob_train, err_prob_test
