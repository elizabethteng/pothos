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
parser.add_argument("--name", help="name of the file being written out, minus filetype",type=str)
args=parser.parse_args()
name=args.name


# import pdspy model data
param_names = ["Tstar","logL_star","logM_disk","logR_disk","h_0","logR_in",\
          "gamma","beta","logM_env","logR_env","f_cav","ksi","loga_max","p","incl"]
dictionary=np.load("./gmd/dictionary.npy")
with open ('./gmd/cubefull.txt', 'rb') as fp:
    cube = np.array(pickle.load(fp))[:,100:500]
with open ('./gmd/cubefull.txt', 'rb') as fp:
    nancube = np.array(pickle.load(fp))[:,100:500]
with open ('./gmd/xvals.txt', 'rb') as fp:
    xvals = pickle.load(fp)[100:500] # normal spaced wavelengths, use np.log10(xvals)

# fix -infs: powerlaw cutoff
for i in range(len(cube)):
    if -np.inf in cube[i]:
        a = cube[i].tolist()
        a.reverse()
        ind = len(a)-a.index(-np.inf)
        x1 = xvals[ind]
        y1 = cube[i][ind]
        for j in range(ind):
            cube[i][j]=(100*(np.log10(xvals[j]/x1)))+y1

# subtracting from each SED its sample mean
nancube[nancube<-20]=np.nan
seds_msub = cube - np.nanmean(nancube,axis=1)[:,np.newaxis]

# run PCA
pca = PCA(n_components=40).fit(seds_msub)
print("PCA finished")
eigenseds=np.array(pca.components_)

# compile PCA weight data
fitdata=[]
for i in range(len(cube)):
    modeldata=[]
    coeffs=pca.transform(seds_msub[i].reshape(1,-1))
    for k in range(15):
        modeldata.append(coeffs[0][k])
    fitdata.append(modeldata)
p=[]
w=[]
m=[]
for i in range(len(cube)):
    pars=[]
    for j in range(len(param_names)):
        if not j in [6,8]:
            pars.append(dictionary[i][param_names[j]])
        if j==6:
            pars.append(np.log10(2.1-1*dictionary[i][param_names[j]]))
        if j==8:
            pars.append(np.log10(-1.5-1*dictionary[i][param_names[j]]))
    p.append(pars)
    weights=[]
    for k in range(15):
        weights.append(fitdata[i][k])
    w.append(weights)
    m.append(np.nanmean(nancube,axis=1)[i])
w=np.transpose(w)
ws=np.ndarray.tolist(w)
ws.append(m)

print("weights calculated")

with open ("./gmd/"+name+"_parvals.txt","wb") as fp:
	pickle.dump(p,fp)

with open ("./gmd/"+name+"_weights.txt","wb") as fp:
	pickle.dump(ws,fp)
    
    
with open ("./gmd/"+name+"_eigenseds.txt","wb") as fp:
	pickle.dump(eigenseds,fp)
with open ("./gmd/"+name+"_mean.txt","wb") as fp:
	pickle.dump(pca.mean_,fp)
    

print("files "+name+" saved")
