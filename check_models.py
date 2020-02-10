#!/usr/bin/env python

import numpy as np
import os

dictionary=np.load("./etgrid/et_dictionary.npy")
for i in range(len(dictionary)):
    if dictionary[i]['filename'] in os.listdir("./etgrid/models"):
        print(i)
