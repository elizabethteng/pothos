#!/usr/bin/env python

import numpy as np
from sklearn.decomposition import PCA
import argparse

def orig_to_trans(pars):
    gamma=pars[6]
    logM_env=pars[8]
    beta=pars[7]
    incl=pars[14]
    
    pars[6]=np.log10(2.1-1*gamma)
    pars[8]=np.log10(-1.5-1*logM_env)
    
    s=np.sin(0.7)
    c=np.cos(0.7)
    
    pars[7]=1-np.cos(((beta*c) + (incl*s/60)-.5)*np.pi/2)
    pars[14]=(-beta*s) + (incl*c/60)
    return pars

param_names = ["Tstar","logLstar","logMdisk","logRdisk","h0","logRin",\
               "gamma","beta","logMenv","logRenv","fcav","ksi","logamax","p","incl"]

seds_dict=np.load("./etgrid/et_dictionary_seds.npy")

seds=np.load("./etgrid/seds.npy")[:,100:500]
nanseds=np.load("./etgrid/seds.npy")[:,100:500]
xvals=np.load("./etgrid/xvals.npy")

# fix -infs: powerlaw cutoff
for i in range(len(seds)):
    if -np.inf in seds[i]:
        a = seds[i].tolist()
        a.reverse()
        ind = len(a)-a.index(-np.inf)
        x1 = xvals[ind]
        y1 = seds[i][ind]
        for j in range(ind):
            seds[i][j]=(100*(np.log10(xvals[j]/x1)))+y1

np.save("./etgrid/sedsflat.npy",np.ndarray.flatten(seds))
            
# subtracting from each SED its sample mean
nanseds[nanseds<-20]=np.nan
seds_msub = seds - np.nanmean(nanseds,axis=1)[:,np.newaxis]


# run PCA
pca = PCA(n_components=40).fit(seds_msub)
print("PCA finished")
eigenseds=np.array(pca.components_)
            
# compile PCA weight data
fitdata=[]
for i in range(len(seds)):
    fitdata.append(pca.transform(seds_msub[0].reshape(1,-1))[0][0:15]) # first 15 weights for each SED
    
params_bypoint=[]
weights_bypoint=[]
means=[]


for i in range(len(seds)):
    pars_orig=[]
    for j in range(len(param_names)):
        pars_orig.append(seds_dict[i][param_names[j]])
    pars_trans=orig_to_trans(pars_orig)
    params_bypoint.append(pars_trans) # coordinates in transformed parameter space
    weights_bypoint.append(fitdata[i])  # set of first 15 weights
    means.append(np.nanmean(nanseds,axis=1)[i])   # mean
    
weights_byweight=np.ndarray.tolist(np.transpose(weights_bypoint))
weights_byweight.append(means)           


print("weights calculated")

parser=argparse.ArgumentParser()
parser.add_argument("--name", help="pca instance nickname",type=str)
args=parser.parse_args()
name=args.name

np.save("./etgrid/"+name+"_coords.npy",params_bypoint)
np.save("./etgrid/"+name+"_eigenseds.npy",eigenseds)
np.save("./etgrid/"+name+"_weights.npy",weights_byweight)
np.save("./etgrid/"+name+"_mean.npy",pca.mean_)

print("files "+name+" saved")
            
