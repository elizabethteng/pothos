#!/usr/bin/env python

import numpy as np
import os
import pickle
from time import time
import pdspy.modeling as modeling
import pdspy.dust as dust

desc='''run new model grid with transformations'''

param_names = ["Tstar","logLstar","logMdisk","logRdisk","h0","logRin",\
          "gamma","beta","logMenv","logRenv","fcav","ksi","logamax","p","incl"]

def run_yso_model( Tstar=None, logL_star=None, logM_disk=None, logR_disk=None, h_0=None, \
                  logR_in=None, gamma=None, beta=None, logM_env=None, logR_env=None, \
                  f_cav=None, ksi=None, loga_max=None, p=None, incl=None):
    
    params = [Tstar,logL_star,logM_disk,logR_disk,h_0,logR_in,\
          gamma,beta,logM_env,logR_env,f_cav,ksi,loga_max,p,incl]
    
    filename=""
    for i in range(len(param_names)):
        filename+=param_names[i]+"_"
        filename+=str(params[i])+"_"
    filename=filename[:-1]
    filename+=".hdf5"
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
    model.add_pringle_disk(mass=10.**logM_disk, rmin=10.**logR_in, \
            rmax=10.**logR_disk, plrho=alpha, h0=h_0, plh=beta, dust=ddust)
    model.add_ulrich_envelope(mass=10.**logM_env, rmin=10.**logR_in, \
            rmax=10.**logR_env, cavpl=ksi, cavrfact=f_cav, dust=edust)
    model.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    # Run the thermal simulation
    model.run_thermal(code="radmc3d", nphot=1e6, \
            modified_random_walk=True, verbose=False, setthreads=9, \
            timelimit=10800)
    
    # Run the SED
    t2=time()
    model.set_camera_wavelength(np.logspace(-1.,4.,500))
    model.run_sed(name="SED", nphot=1e5, loadlambda=True, incl=incl, \
            pa=0., dpc=140., code="radmc3d", camera_scatsrc_allfreq=True, \
            verbose=False, setthreads=9)

    print("finished running SED for "+filename[0:40]+"... in %0.3fs" % (time() - t2))
    
    # Write out the file.
    model.write_yso("./etgrid/filename/"+filename)

with open ('./etgrid/etgrid_coords_bypoint_orig.txt', 'rb') as fp:
    coords = pickle.load(fp)    
    
for i in range(1500,2000):
    t0=time()
    run_yso_model(Tstar=coords[i][0], logL_star=coords[i][1], logM_disk=coords[i][2], \
                  logR_disk=coords[i][3], h_0=coords[i][4], logR_in=coords[i][5],\
                  gamma=coords[i][6], beta=coords[i][7], logM_env=coords[i][8], \
                  logR_env=coords[i][9], f_cav=coords[i][10], ksi=coords[i][11], \
                  loga_max=coords[i][12], p=coords[i][13], incl=coords[i][14])
    print("finished running SED #"+str(i)+" in %0.3fs" % (time() - t0))