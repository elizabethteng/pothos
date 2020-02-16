#!/usr/bin/env python

import numpy as np
import os
import pickle
from time import time
import pdspy.modeling as modeling
from pyDOE import lhs

param_names = ["Tstar","logLstar","logMdisk","logRdisk","h0","logRin",\
          "gamma","beta","logMenv","logRenv","fcav","ksi","logamax","p","incl"]
ranges = [[3000.,5000.], [-1,3.],[-8.,-2.], [0.,3.],[0.01,0.5], [-1.,2.5], [0.0,1.999], \
          [0.5,2.0],[-8.,-2.],[2.5,4.], [0.,1.], [0.5,1.5], [0.,5.], [2.5,4.5], [0.,90.]]

new_param_names = ["T_star","logL_star","logM_disk","logR_disk","h_0","logR_in",\
               "gamma_trans","bi_x","logM_env_trans","logR_env","f_cav","ksi","loga_max","p","bi_y"]
new_ranges = [[3000.,5000.], [-1,3.],[-8.,-2.], [0.,3.],[0.01,0.5], [-1.,2.5], [np.log10(0.101),np.log10(2.1)], \
          [0,2],[np.log10(0.5),np.log10(6.5)],[2.5,4.], [0.,1.], [0.5,1.5], [0.,5.], [2.5,4.5], [-1.25,0.75]]

def generate_ranges(ranges,scale_factors):
    new_ranges=[]
    for i in range(len(ranges)):
        if scale_factors[i]==1:
            new_ranges.append(ranges[i])
        else:
            a=ranges[i][0]
            b=ranges[i][1]
            s=scale_factors[i]
            new_ranges.append([(a-(s/2-0.5)*(b-a)),(b+(s/2-0.5)*(b-a))])
    return (new_ranges)

scaleup=[4,1.5,1,2,1,1,1,1,1,3,2,1,1,1.8,1]
inflated_ranges=generate_ranges(new_ranges,scaleup)

totalscale=1.5 # for beta/incl cut
for i in range(len(scaleup)):
    totalscale=totalscale*scaleup[i]
print(totalscale*4000)

n_models=int(4000*totalscale)
ranges=inflated_ranges

dimensions = len(ranges)
ndsample_01 = np.array(lhs(dimensions, samples=n_models))
print("sample successfully drawn")

scale = []
offset = []
for i in range(len(ranges)):
    scale.append(ranges[i][1]-ranges[i][0])
    offset.append(ranges[i][0])

ndsample_toscale = ndsample_01 * np.array(scale)
ndsample_inplace = ndsample_toscale + np.array(offset)

sample_bypoint = ndsample_inplace
sample_byaxis = np.transpose(sample_bypoint)

with open('./etgrid/raw_coords_byaxis.txt', 'wb') as fp:
    pickle.dump(sample_byaxis, fp)
with open('./etgrid/raw_coords_bypoint.txt', 'wb') as fp:
    pickle.dump(sample_bypoint, fp)
    
    