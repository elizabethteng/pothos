#!/usr/bin/env python

import numpy as np
import pickle


with open ('./etgrid/raw_coords_bypoint.txt', 'rb') as fp:
    raw_coords_bypoint = pickle.load(fp)
