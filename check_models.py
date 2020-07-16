#!/software/et_env/bin python

import numpy as np
import os

dictionary=np.load("./data/pothos/etgrid/et_dictionary.npy")

finished=[]

for i in range(len(dictionary)):
    if dictionary[i]['filename'] in os.listdir("./data/pothos/etgrid/models"):
        finished.append(i)

report=[]
report.append(finished[0])

for i in range(1,len(finished)):
    if int(finished[i])!=int(finished[i-1])+1:
        report.append(finished[i-1])
        report.append(finished[i])
report.append(finished[-1])

for i in range(int(len(report)/2)):
    print(report[2*i],report[2*i+1])

print(len(finished))
