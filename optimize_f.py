#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import pickle
from time import time
from sklearn.decomposition import PCA
import george
from george import kernels
from scipy.optimize import minimize
import sys



with open ("./gmd/cubeflat.txt","rb") as fp:
	cubeflat=pickle.load(fp)
name=str(sys.argv[1])
with open ("./gmd/"+name+"_parvals.txt","rb") as fp:
	Xs=pickle.load(fp)
with open ("./gmd/"+name+"_weights.txt","rb") as fp:
	ws=pickle.load(fp)

yerrs=[]
for i in range(16):
	yerrs.append([x*0.01 for x in ws[i]])


initvecs=[]
for i in range(16):
    initvecs.append([ 6.33043185, 18.42068074,  0.        ,  0.        , -0.4462871 ,
       -5.05145729, -1.38629436,  1.7       , -3.21887582,  0.        ,
       -3.21887582, -2.77258872, -2.77258872,  1.38629436,  2.19722458,
        3.21887582])

kernel = 23*kernels.ExpSquaredKernel(1**2,ndim=15,axes=0)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=1)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=2)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=3)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=4)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=5)*\
        kernels.PolynomialKernel(1**2,15,ndim=15,axes=6)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=7)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=8)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=9)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=10)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=11)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=12)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=13)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=14) 
blankhodlr=george.GP(kernel,solver=george.HODLRSolver)

def F(hyperparams,gp):
    t0=time()
    preds=[]
    for i in range(len(ws)):  # same covfunc for each weight and the sample mean
        t1=time()
        gp.set_parameter_vector(hyperparams[i])
        gp.compute(Xs,yerrs[i])
        print("GP computed in %0.3fs" % (time() - t1))
        t2=time()
        pred, pred_var = gp.predict(ws[i], Xs, return_var=True)
        preds.append(pred)
        print("predictions made in %0.3fs" % (time() - t2))    
    reconst_SEDs=[]
    for i in range(3850):
        reconst=np.dot(np.array(preds)[:,i][0:15],eigenseds[0:15]) + pca.mean_ + np.array(preds)[:,i][15]
        reconst_SEDs.append(reconst)
    print("done in %0.3fs" % (time() - t0))
    return reconst_SEDs

allseds=F(initvecs,blankhodlr)



