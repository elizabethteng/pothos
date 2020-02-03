#!/usr/bin/env python

import numpy as np
import os
import pickle
from time import time
import pdspy.modeling as modeling

with open ('./etgrid/etgrid_coords_byaxis_orig.txt', 'rb') as fp:
    coords_byaxis = pickle.load(fp)
with open ('./etgrid/etgrid_coords_bypoint_orig.txt', 'rb') as fp:
    coords_bypoint = pickle.load(fp)
    
    
directory="./etgrid/models/"
coords=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]] # by axis, orig
seds=[]
for i in range(len(os.listdir(directory))):
    for j in range(len(coords)-1):
        coords[j].append(float(os.listdir(directory)[i].split("_")[(2*j)+1])
    coords[14].append(float(os.listdir(directory)[i].split("_")[(2*14)+1][:-5]))
    filename=os.listdir(directory)[i]
    model=modeling.YSOModel()
    model.read_yso(directory+filename)
    seds.append(np.log10(model.spectra["SED"].flux))
                         
with open('./etgrid/etgrid_seds.txt', 'wb') as fp:
    pickle.dump(seds, fp)
with open('./grid_metadata/etgrid_coords_ordered.txt', 'wb') as fp:
    pickle.dump(coords, fp)