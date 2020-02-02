#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import pickle
from time import time
from sklearn.decomposition import PCA
import george
from george import kernels
from scipy.optimize import minimize
import argparse

desc='''
Optimize F -- optimizes 15d GP hyperparameters, minimizing chisq between reconst library and pdspy library.
'''
parser=argparse.ArgumentParser(description=desc)
parser.add_argument("--rname", help="name of the pca weights read in for training data",type=str)
parser.add_argument("--wname", help="name of the results being written out, minus filetype",type=str)
parser.add_argument("--verbose",help="print out chisq and hyperparameter vector in the likelihood function",action="store_true")
args=parser.parse_args()
rname=args.rname
wname=args.wname

# load cube
with open ("./gmd/cubeflat.txt","rb") as fp:
	cubeflat=pickle.load(fp)

# load pca weight data generated by get_pca_weights.py
with open ("./gmd/"+rname+"_parvals.txt","rb") as fp:
	Xs=pickle.load(fp)
with open ("./gmd/"+rname+"_weights.txt","rb") as fp:
	ws=pickle.load(fp)
with open ("./gmd/"+rname+"_eigenseds.txt","rb") as fp:
	eigenseds=pickle.load(fp)
with open ("./gmd/"+rname+"_mean.txt","rb") as fp:
	pcamean=pickle.load(fp)
print("pca weights loaded from "+rname)

yerrs=[]
for i in range(16):
	yerrs.append([x*0.05 for x in ws[i]])
initvecs=[]
for i in range(16):
    initvecs.append([ 6.33043185, 18.42068074,  0.,  0., -0.4462871 , -5.05145729, -1.38629436, 
                     -2.4079456086518722, -2.4079456086518722,  -3.2188758248682006, -3.21887582, 
                     -2.77258872, -2.77258872,  1.38629436,  2.19722458, 0.8109302162163288])


kernel = 23*kernels.ExpSquaredKernel(1**2,ndim=15,axes=0)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=1)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=2)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=3)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=4)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=5)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=6)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=7)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=8)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=9)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=10)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=11)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=12)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=13)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=14) 
blankhodlr=george.GP(kernel,solver=george.HODLRSolver)

def F_chisq_quiet(hp,gp):
    t0=time()
    hyperparams=np.transpose(np.array(hp).reshape(16,16))
    print(hyperparams)
    preds=[]
    for i in range(len(ws)):  # same covfunc for each weight and the sample mean
        t1=time()
        gp.set_parameter_vector(hyperparams[i])
        if args.verbose==True:
            print("weight #"+str(i))
            print(gp.get_parameter_vector())
        gp.compute(Xs,yerrs[i])
        t2=time()
        pred, pred_var = gp.predict(ws[i], Xs, return_var=True)
        preds.append(pred)
    reconst_SEDs=[]
    for i in range(3850):
        reconst=np.dot(np.array(preds)[:,i][0:15],eigenseds[0:15]) + pcamean + np.array(preds)[:,i][15]
        reconst_SEDs.append(reconst)
    allsedsflat=np.ndarray.flatten(np.array(reconst_SEDs))
    chisq=np.sum((cubeflat-allsedsflat)**2/0.1)
    print(chisq)
    print(time()-t0)
    return chisq

def chisq(p):
    return F_chisq_quiet(p,blankhodlr)

print("starting minimize routine")
t0=time()
result = minimize(chisq,initvecs,method="COBYLA")
print("minimize routine done in %0.3fs" % (time() - t0))

print("Final chisq: "+np.array(result.x).reshape(16,16))
with open ("./"+wname+"_optimize_result.txt","wb") as fp:
	pickle.dump(np.array(result.x).reshape(16,16),fp)
with open ("./"+wname+"_time_rec.txt","wb") as fp:
	pickle.dump(str(time() - t0),fp)
