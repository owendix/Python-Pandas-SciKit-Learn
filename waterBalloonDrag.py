# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:21:02 2015

@author: owendix
"""

from scipy.integrate import ode
import scipy as sp
import numpy as np
from pandas import DataFrame
import pandas as pd
import os.path


def xHitNoDrag(bandLs,unStretchL,k,g,ms,c,mL,isEgChanging=True):
    
    #y-intercept from fit is just unStretchL
    xi = bandLs - unStretchL
    if isEgChanging:
        hf = sp.sqrt(2*(bandLs**2 - unStretchL**2))
    else:
        hf = sp.zeros(bandLs.shape)      
    
    d = ((k*xi**2/g).T - hf.T.dot(ms + c*mL))/(ms+c**2*mL)   
    
    #must be done to use as columns and index
    ms = sp.squeeze(ms)
    bandLs = sp.squeeze(bandLs)    
    
    xHit = DataFrame(d,columns=ms,index=bandLs)
    xHit.columns.name = 'mass(kg)'
    xHit.index.name = 'bandL(m)'  
    
    return xHit


def xHitDrag(bandLs,unStretchL,k,g,ms,c,mL,B,isEgChanging=True,
             printVLaunch=False,printAllStats=False):

    theta = 45
    theta *= (sp.pi/180)
    consts = [ms[0,0],B,g]
    
    def f(t, y, consts):
        m = consts[0]
        B = consts[1]
        g = consts[2]
        vtemp = -B*m**(-1/3)*sp.sqrt(y[0]**2 + y[1]**2)
        #Y' = f(t,Y) according to python's info: scipy.integrate.ode?
        #Y = [vx, vy, x, y]
        #Y' = [ax, ay, vx, vy]
        return [vtemp*y[0], -g - vtemp*y[1], y[0], y[1]]
    
    def jac(t, y, consts):
        m = consts[0]
        B = consts[1]
        jconst = -B*m**(-1/3)/sp.sqrt(y[0]**2+y[1]**2)
        j12 = jconst*y[0]*y[1]
        j11 = jconst*y[0]**2
        j22 = jconst*y[1]**2
        #jac = df/dY according to python's info: scipy.integrate.ode?
        return [[j11,j12,0,0], [j12,j22,0,0], [1,0,0,0], [0,1,0,0]]
    
    def findImpact(y0, t0, consts):
        r = ode(f, jac).set_integrator('lsoda')
        r.set_initial_value(y0, t0).set_f_params(consts).set_jac_params(consts)
    
        t1 = 10
        dt = 0.01
        while r.successful() and r.t < t1 and r.y[3] >= 0.0:
            r.integrate(r.t+dt)
    #        print('%2.2fs: v = (%2.2f, %2.2f)m/s, r = (%2.2f, %2.2f)m' % (r.t, r.y[0],r.y[1],r.y[2],r.y[3]))
            
        #return first negative term
        return r.t, r.y
    
    
    """
    Broadcasting rule: Two arrays are compatible for broadcasting if,
    for each trailing dimension (starting from the end), the axis lengths
    match or if either of the lengths is 1. Broadcasting is performed over the
    missing and/or length 1 dimensions.
    """
    
    #y-intercept from fit is just unStretchL
    xi = bandLs - unStretchL
    if isEgChanging:
        hf = sp.sqrt(2*(bandLs**2 - unStretchL**2))
    else:
        hf = sp.zeros(bandLs.shape)
    
    vf = sp.sqrt(((k*xi**2).T - g*hf.T.dot(ms + c*mL))/(ms+c**2*mL))
    
    ms = sp.squeeze(ms)
    bandLs = sp.squeeze(bandLs)
    #Create Series and DataFrame structures
    if printAllStats:
        vLaunch = DataFrame(vf,columns=ms,index=bandLs)    
        tHit = DataFrame(columns=ms,index=bandLs)
        vxHit = DataFrame(columns=ms,index=bandLs)
        vyHit = DataFrame(columns=ms,index=bandLs)
        yHit = DataFrame(columns=ms,index=bandLs)
        vLaunch.columns.name = 'mass(kg)'
        vLaunch.index.name = 'bandL(m)'
        tHit.columns.name = 'mass(kg)'
        tHit.index.name = 'bandL(m)'
        vxHit.columns.name = 'mass(kg)'
        vxHit.index.name = 'bandL(m)'
        vyHit.columns.name = 'mass(kg)'
        vyHit.index.name = 'bandL(m)'
        yHit.columns.name = 'mass(kg)'
        yHit.index.name = 'bandL(m)'
    elif printVLaunch:
        vLaunch = DataFrame(vf,columns=ms,index=bandLs)
        vLaunch.columns.name = 'mass(kg)'
        vLaunch.index.name = 'bandL(m)'
        
    xHit = DataFrame(columns=ms,index=bandLs)
    xHit.columns.name = 'mass(kg)'
    xHit.index.name = 'bandL(m)'
    
    
    #ndimensional version of enumerate
    for (il, im), v0 in sp.ndenumerate(vf):
    
        vx0 = v0*sp.cos(theta)
        vy0 = v0*sp.sin(theta)
        
        y0, t0 = [vx0, vy0, 0.0, 0.0], 0.0
        
        consts[0] = ms[im]
        
        t_hit, y_hit = findImpact(y0,t0,consts)
        
        L = bandLs[il]
        M = ms[im]
        
        if printAllStats:
            tHit.set_value(L,M, t_hit)
            vxHit.set_value(L,M, y_hit[0])
            vyHit.set_value(L,M, y_hit[1])
            yHit.set_value(L,M, y_hit[3])
        
        #always print xHit
        xHit.set_value(L,M, y_hit[2])
    
    #only display 2 values past the decimal
    pd.options.display.float_format = '{:,.2f}'.format
    if printVLaunch or printAllStats:
        vLaunch.to_csv(fout,sep='\t')
        print('vLaunch (m/s)',vLaunch,sep='\n')
    if printAllStats:
        print('tHit (s)',tHit,sep='\n')
        print('vxHit (m/s)',vxHit,sep='\n')
        print('vyHit (m/s)',vyHit,sep='\n')
        print('yHit (m)',yHit,sep='\n')
        print('xHit (m)',xHit,sep='\n')
    
    return xHit

printAllStats = False
printVLaunch = False

#if false, do not account for change in Eg in launcher
isEgChanging = True

if printVLaunch:
    fout = 'wBData'
    if isEgChanging:
        fout += 'WithDeltaEg'
    else:
        fout += 'NoDeltaEg'
    
    fout += '.txt'

#determined by fit of F vs L
unStretchL = 0.7173
k = 91.68 #k = 103.93  #N/m of WBlauncher 1

mL = 0.188 #mL = 0.1523 #kg mass of launcher
c = 0.523   #center of mass of launcher
g = 9.81
B = 0.003025    #(kg^(1/3)/m) for drag
#theta = 45

#range of masses and stretch distances
ms = sp.arange(0.15,0.205,0.01)[sp.newaxis,:]
#zis = sp.arange(0.75,1.355,0.03)[sp.newaxis,:]
bandLs = sp.arange(1.0,2.01,0.05)[sp.newaxis,:]

xHits = dict()
keys=['mLDragEg','mLDrag','mLEg','mL','DragEg','Drag','Eg','Simplest']

xHits[keys[0]]=xHitDrag(bandLs,unStretchL,k,g,ms,c,mL,B,isEgChanging=True,
             printVLaunch=False,printAllStats=False
             )
xHits[keys[1]]=xHitDrag(bandLs,unStretchL,k,g,ms,c,mL,B,isEgChanging=False,
             printVLaunch=False,printAllStats=False
             )
xHits[keys[2]]=xHitNoDrag(bandLs,unStretchL,k,g,ms,c,mL,isEgChanging=True)
xHits[keys[3]]=xHitNoDrag(bandLs,unStretchL,k,g,ms,c,mL,isEgChanging=False)
mL = 0.
xHits[keys[4]]=xHitDrag(bandLs,unStretchL,k,g,ms,c,mL,B,isEgChanging=True,
             printVLaunch=False,printAllStats=False
             )
xHits[keys[5]]=xHitDrag(bandLs,unStretchL,k,g,ms,c,mL,B,isEgChanging=False,
             printVLaunch=False,printAllStats=False
             )
xHits[keys[6]]=xHitNoDrag(bandLs,unStretchL,k,g,ms,c,mL,isEgChanging=True)
xHits[keys[7]]=xHitNoDrag(bandLs,unStretchL,k,g,ms,c,mL,isEgChanging=False)    

#append
fname = 'wbData.txt'
try:
    os.remove(fname)
except OSError:
    pass

with open(fname, 'a') as f:

    for k in keys:
        f.write('Details Included: '+k+'\n')
        f.write('xHit (m)\n')
        xH = np.round(xHits[k],2)
        xH.to_csv(f,sep='\t',float_format='%.2f',mode='a')
        f.write('\n')
        
    f.write('Differences between predictions\n')

    for i,k in enumerate(keys):
        for j in range(i+1,len(keys)):
            k2 = keys[j]
            f.write('Difference in xHits: '+k+' - '+k2+'\n')
            xDiff = xHits[k] - xHits[k2]
            xD = np.round(xDiff,2)
            xD.to_csv(f,sep='\t',float_format='%.2f',mode='a')
            f.write('\n')
