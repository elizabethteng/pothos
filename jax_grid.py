#!/usr/bin/env python

import numpy as np
import os
import pickle
from time import time
import pdspy.modeling as modeling
import pdspy.dust as dust
import argparse

d
parser=argparse.ArgumentParser()
parser.add_argument("--threads", help="number of threads",type=int)
parser.add_argument("--start",help="starting number",type=int)

param_names = ["Tstar","logLstar","logMdisk","logRdisk","h0","logRin",\
          "gamma","beta","logMenv","logRenv","fcav","ksi","logamax","p","incl"]

def run_yso_model( Tstar=None, logL_star=None, logM_disk=None, logR_disk=None, h_0=None, \
                  logR_in=None, gamma=None, beta=None, logM_env=None, logR_env=None, \
                  f_cav=None, ksi=None, loga_max=None, p=None, incl=None, filename=None):
    
    print("starting on "+filename)
  
    # Set up the dust properties.
    dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/diana_wice.hdf5")
    ddust = dust_gen(10.**loga_max / 1e4, p)
    env_dust_gen = dust.DustGenerator(dust.__path__[0]+\
            "/data/diana_wice.hdf5")
    edust = env_dust_gen(1.0e-4, 3.5)

    # Calculate alpha correctly.
    alpha = gamma + beta

    # Fix the scale height of the disk.
    h_0 *= (10.**logR_disk)**beta

    # Set up the model.
    model = modeling.YSOModel()
    model.add_star(luminosity=10.**logL_star, temperature=Tstar)
    model.set_spherical_grid(10.**logR_in, 10.**logR_env, 100, 101, 2, \
            code="radmc3d")
    model.add_pringle_disk(mass=10.**logM_disk, rmin=10.**logR_in,\
            rmax=10.**logR_disk, plrho=alpha, h0=h_0, plh=beta, dust=ddust)
    model.add_ulrich_envelope(mass=10.**logM_env, rmin=10.**logR_in, \
            rmax=10.**logR_env, cavpl=ksi, cavrfact=f_cav, dust=edust)
    model.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    # Run the thermal simulation
    model.run_thermal(code="radmc3d", nphot=1e6, \
            modified_random_walk=True, verbose=False, setthreads=18, \
            timelimit=10800)
    
    # Run the SED
    model.set_camera_wavelength(np.logspace(-1.,4.,500))
    model.run_sed(name="SED", nphot=1e5, loadlambda=True, incl=incl, \
            pa=0., dpc=140., code="radmc3d", camera_scatsrc_allfreq=True, \
            verbose=False, setthreads=18)
    
    # Write out the file.
    model.write_yso("./etgrid/models/"+filename)

dictionary=np.load("./etgrid/et_dictionary.npy")
    
for i in range(26,700):
    try:
        t0=time()
        point=dictionary[i]
        run_yso_model( Tstar=point["Tstar"], logL_star=point["logLstar"], logM_disk=point["logMdisk"], \
                      logR_disk=point["logRdisk"], h_0=point["h0"], logR_in=point["logRin"], gamma=point["gamma"], \
                      beta=point["beta"], logM_env=point["logMenv"], logR_env=point['logRenv'], \
                      f_cav=point['fcav'], ksi=point['ksi'], loga_max=point['logamax'], p=point['p'], \
                      incl=point['incl'], filename=point['filename'])
        print("finished running SED #"+str(i)+" in %0.3fs" % (time() - t0))
    except:
        print(str(i)+" timed out")
        pass
