#!/usr/bin/env python

import numpy as np
import pickle

new_ranges = [[3000.,5000.], [-1,3.],[-8.,-2.], [0.,3.],[0.01,0.5], [-1.,2.5], [np.log10(0.101),np.log10(2.1)], \
          [0,2],[np.log10(0.5),np.log10(6.5)],[2.5,4.], [0.,1.], [0.5,1.5], [0.,5.], [2.5,4.5], [-1.25,0.75]]

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

with open ('./etgrid/raw_coords_bypoint.txt', 'rb') as fp:
    raw_coords_bypoint = pickle.load(fp)
    
good_trans=[]
good_orig=[]
    
for i in range(len(raw_coords_bypoint)):
    point=raw_coords_bypoint[i]
    add=True
    for j in (0,1,3,9,10,13):
        if not new_ranges[j][0]<=point[j]<=new_ranges[j][1]:
            add=False
    origpoint=trans_to_orig(point)
    if not 0.5<=origpoint[7]<=2.0:
        add=False
    if not 0<=origpoint[14]<=90:
        add=False
    if add==True:
        good_trans.append(point)
        good_orig.append(origpoint)
        
print("number of good points:",str(len(good_trans)))

with open('./etgrid/etgrid_coords_byaxis_trans.txt', 'wb') as fp:
    pickle.dump(np.transpose(np.array(good_trans)), fp)
with open('./etgrid/etgrid_coords_bypoint_trans.txt', 'wb') as fp:
    pickle.dump(np.array(good_trans), fp)
    
with open('./etgrid/etgrid_coords_byaxis_orig.txt', 'wb') as fp:
    pickle.dump(np.transpose(np.array(good_orig)), fp)
with open('./etgrid/etgrid_coords_bypoint_orig.txt', 'wb') as fp:
    pickle.dump(np.array(good_orig), fp)


