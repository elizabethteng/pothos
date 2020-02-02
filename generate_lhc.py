#!/usr/bin/env python

import lhsmdu
from time import time
import numpy as np
import pickle

def generate_lhc(n_models, ranges): 
    dimensions = len(ranges)
    ndsample_01 = np.array(lhsmdu.sample(dimensions, n_models))
    scale = []
    offset = []
    for i in range(len(ranges)):
        scale.append(ranges[i][1]-ranges[i][0])
        offset.append(ranges[i][0])

    ndsample_toscale = ndsample_01 * np.array(scale)[:,np.newaxis] 
    ndsample_inplace = ndsample_toscale + np.array(offset)[:,np.newaxis]

    sample_byaxis = ndsample_inplace
    sample_bypoint = np.transpose(sample_byaxis)
    
    return (sample_byaxis, sample_bypoint)

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


new_ranges = [[3000.,5000.], [-1,3.],[-8.,-2.], [0.,3.],[0.01,0.5], [-1.,2.5], [np.log10(0.101),np.log10(2.1)], \
          [0,2],[np.log10(0.5),np.log10(6.5)],[2.5,4.], [0.,1.], [0.5,1.5], [0.,5.], [2.5,4.5], [-1.25,0.75]]

scaleup=[4,1.5,1,2,1,1,1,1,1,3,2,1,1,1.8,1]
inflated_ranges=generate_ranges(new_ranges,scaleup)

totalscale=1.5 # for beta/incl cut
for i in range(len(scaleup)):
    totalscale=totalscale*scaleup[i]

t0=time()
coords_byaxis,coords_bypoint = generate_lhc(int(4000*totalscale),inflated_ranges)
print(time()-t0)

with open('./etgrid/raw_coords_byaxis.txt', 'wb') as fp:
    pickle.dump(coords_byaxis, fp)
with open('./etgrid/raw_coords_bypoint.txt', 'wb') as fp:
    pickle.dump(coords_bypoint, fp)


#with open('./etgrid/etgrid_coords_byaxis_trans.txt', 'wb') as fp:
#    pickle.dump(good_coords_byaxis, fp)
#with open('./etgrid/etgrid_coords_bypoint_trans.txt', 'wb') as fp:
#    pickle.dump(good_coords_bypoint, fp)


