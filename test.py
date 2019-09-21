#!/usr/bin/env python

import sys
import pickle
import numpy as np

a=[0,1,2,3,4,5]

print("hello")

name=str(sys.argv[1])

#with open("./"+name+".txt","wb") as fp:

#	pickle.dump(a,fp)

with open ("./"+name+".txt","rb") as fp:
    b=pickle.load(fp)

print("b ", np.array(b)+1)
