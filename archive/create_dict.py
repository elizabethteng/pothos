#!/usr/bin/env python

import numpy as np
import pickle


with open ('./etgrid/etgrid_coords_bypoint_orig.txt', 'rb') as fp:
    coords_bypoint = pickle.load(fp)
    
with open ('./etgrid/etgrid_coords_bypoint_orig.txt', 'rb') as fp:
    coords_bypoint_copy = pickle.load(fp)
    
    
filenames=[]
param_names = ["Tstar","logLstar","logMdisk","logRdisk","h0","logRin",\
          "gamma","beta","logMenv","logRenv","fcav","ksi","logamax","p","incl"]

for i in range(len(coords_bypoint)):
    filename=""
    params=np.round(coords_bypoint[i],3)
    for i in range(len(param_names)):
        filename+=param_names[i]+"_"
        filename+=str(params[i])+"_"
    filename=filename[:-1]
    filename+=".hdf5"
    filenames.append(filename)
    
    
dictionary=[]

for i in range(len(coords_bypoint_copy)):
    pointdict={}
    for j in range(15):
        pointdict[param_names[j]]=coords_bypoint_copy[i][j]
    pointdict["filename"]=filenames[i]
    dictionary.append(pointdict)
    
np.save("./etgrid/et_dictionary.npy",np.array(dictionary))