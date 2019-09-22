#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import pickle
from time import time
from sklearn.decomposition import PCA
import george
from george import kernels
from scipy.optimize import minimize
import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
import argparse

desc='''
Figure factory, produces different plots and animations for my talks.
'''
parser=argparse.ArgumentParser(description=desc)

parser.add_argument("--which", help="the type of figure being made",type=str, default="testfig")
parser.add_argument("--name", help="name of the figure, minus filetype",type=str)
parser.add_argument("--cubeind", help="pdspy model number",type=int)


args=parser.parse_args()

# import pdspy model data
param_names = ["Tstar","logL_star","logM_disk","logR_disk","h_0","logR_in",\
          "gamma","beta","logM_env","logR_env","f_cav","ksi","loga_max","p","incl"]
dictionary=np.load("./gmd/dictionary.npy")
with open ('./gmd/cubefull.txt', 'rb') as fp:
    cube = np.array(pickle.load(fp))[:,100:500]
with open ('./gmd/xvals.txt', 'rb') as fp:
    xvals = pickle.load(fp)[100:500] # normal spaced wavelengths, use np.log10(xvals)
for i in range(len(cube)):
    if -np.inf in cube[i]:
        a = cube[i].tolist()
        a.reverse()
        ind = len(a)-a.index(-np.inf)
        x1 = xvals[ind]
        y1 = cube[i][ind]
        for j in range(ind):
            cube[i][j]=(100*(np.log10(xvals[j]/x1)))+y1

# figures
if args.which=="testfig":
	plt.figure(figsize=(12,10))
	plt.plot(np.log10(xvals),cube[args.cubeind])
	plt.xlabel("log(Wavelength) - microns").set_fontsize(16)
	plt.ylabel("log(Flux Density) - Janskys").set_fontsize(16)
	plt.tick_params(labelsize=12)
	plt.title("example SED generated by pdspy",pad=20).set_fontsize(20)
	plt.savefig("./"+args.name+".png")	
	print("./"+args.name+".png saved")

