#!/usr/bin/env python

import numpy as np
import pickle

def trans_to_orig(pars):
    gamma_trans=pars[6]
    logM_env_trans=pars[8]
    bi_x=pars[7]
    bi_y=pars[14]
    
    pars[6]=2.1 - 10**gamma_trans
    pars[8]=-1.5 - 10**logM_env_trans
    
    s=np.sin(0.7)
    c=np.cos(0.7)
    
    pars[7]=round((1/(c+(s**2/c)))*((2/np.pi)*np.arccos(1-bi_x)+0.5-(s/c)*bi_y),14)
    pars[14]=round((60*s)*((2/np.pi)*np.arccos(1-bi_x)+0.5+(c/s)*bi_y),14)
    return pars

def orig_to_trans(pars):
    gamma=pars[6]
    logM_env=pars[8]
    beta=pars[7]
    incl=pars[14]
    pars[6]=np.log10(2.1-1*gamma)
    pars[8]=np.log10(-1.5-1*logM_env)
    s=np.sin(0.7)
    c=np.cos(0.7)
    theta=0.7
    pars[7]=1-np.cos(((beta*c)+(incl*s/60)-0.5)*np.pi/2)
    pars[14]=(-beta*s)+(incl*c/60)
    return pars

with open ('./etgrid/etgrid_coords_bypoint_trans.txt', 'rb') as fp:
    raw_coords_bypoint = pickle.load(fp)
    
good_trans=[]
good_orig=[]
    
for i in range(len(raw_coords_bypoint)):
    point=raw_coords_bypoint[i]
    origpointfull=trans_to_orig(point)
    origpoint=[]
    #if round(origpointfull[8],4)==0:
    print("oops! ",i," is logmenv= ",origpointfull[8])
    rounds=[2,4,4,4,5,4,4,4,4,4,5,5,4,4,3]
    for j in range(15):
        origpoint.append(round(origpointfull[j],rounds[j]))    
    good_trans.append(orig_to_trans(origpoint))
    good_orig.append(origpoint)
        
print("number of good points:",str(len(good_trans)))

#with open('./etgrid/etgrid_coords_byaxis_trans.txt', 'wb') as fp:
#    pickle.dump(np.transpose(np.array(good_trans)), fp)
#with open('./etgrid/etgrid_coords_bypoint_trans.txt', 'wb') as fp:
#    pickle.dump(np.array(good_trans), fp)
    
#with open('./etgrid/etgrid_coords_byaxis_orig.txt', 'wb') as fp:
#    pickle.dump(np.transpose(np.array(good_orig)), fp)
#with open('./etgrid/etgrid_coords_bypoint_orig.txt', 'wb') as fp:
#    pickle.dump(np.array(good_orig), fp)


