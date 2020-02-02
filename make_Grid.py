#!/usr/bin/env python3

from pdspy.modeling import YSOModel
from pdspy.dust import Dust
from os.path import exists
from datetime import datetime

# Star parameters

T_star = [3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000, \
          9500,10000]

L_star = [0.1,0.5,1,2,3,4,5,6,7,8,9,10]

# Disk parameters:

M_disk = [1.0e-6,2.5e-6,5.0e-6,7.5e-6,1.0e-5,2.5e-5,5.0e-5,7.5e-5, \
          1.0e-4,2.5e-4,5.0e-4,7.5e-4,1.0e-3,2.5e-3,5.0e-3,7.5e-3,1.0e-2]

R_disk = [10,50,100,150,200,250,300,350,400,450,500]

h_0 = [0.05,0.1,0.15,0.2]

R_in = 0.1

beta = 58./45

alpha = 3*(beta-0.5)

# Envelope parameters:

M_env = [1.0e-6,2.5e-6,5.0e-6,7.5e-6,1.0e-5,2.5e-5,5.0e-5,7.5e-5, \
          1.0e-4,2.5e-4,5.0e-4,7.5e-4,1.0e-3,2.5e-3,5.0e-3,7.5e-3,1.0e-2]

R_env = [100,300,500,1000,1500,2000]

R_c = R_disk

f_cav = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

ksi = [0.8,1.0,1.2]

# Dust parameters:

dustopac = ["pollack_1um.hdf5","pollack_10um.hdf5","pollack_100um.hdf5", \
        "pollack_1mm.hdf5", "pollack_1cm.hdf5","pollack_10cm.hdf5"]

# Create the grid of models:

count = 0

for i in [2]:
 for j in [2,4,7,11]:
  for k in [2,4,6,8,10,12,14,16]:
   for l in [1,2,5,10]:
    for m in [0,2]:
     for n in [2,4,6,8,10,12,14,16]:
      for o in [1,2,3,5]:
       if (R_env[o] < R_disk[l]):
        continue
       for p in [0,2,6,11]:
        for q in [1]:
         for r in [0,3]:
             filename = "{0:02d}-{1:02d}-{2:02d}-{3:02d}-{4:02d}-{5:02d}-" \
                     "{6:02d}-{7:02d}-{8:02d}-{9:02d}.rt".format(i,j,k,l,m,n, \
                     o,p,q,r)

             if exists(filename):
                 print("\nSkipping model "+filename+ \
                         " because it already exists.\n")
             else:
                 if (datetime.weekday(datetime.now()) < 5):
                     if (datetime.now().hour >= 9) and \
                             (datetime.now().hour < 17):
                         nprocesses = 5
                     else:
                         nprocesses = 7

                 run_yso_model(filename, Tstar=Tstar[i], Lstar=Lstar[j], \
                         M_disk=M_disk[k], R_disk=R_disk[l], h_0=h_0[m], \
                         R_in=R_in, beta=beta, alpha=alpha, M_env=M_env[n], \
                         R_env=R_env[o], R_c=R_disk[l], f_cav=f_cav[p], \
                         ksi=ksi[q], disk_dust=dustopac[r], \
                         envelope_dust=dustopac[0], nprocesses=nprocesses)

def run_yso_model(filename, Tstar=None, Lstar=None, M_disk=None, R_disk=None, \
    h_0=None, R_in=None, beta=None, alpha=None, M_env=None, R_env=None, \
    R_c=None, f_cav=None, ksi=None, disk_dust=None, envelope_dust=None, \
    nprocesses=None):

    ddust = Dust()
    ddust.set_properties_from_file("../Dust/"+disk_dust)

    edust = Dust()
    edust.set_properties_from_file("../Dust/"+envelope_dust)

    model = YSOModel()
    model.add_star(luminosity=Lstar, temperature=Tstar)
    model.set_spherical_grid(R_in, R_env, 100, 201, 2, code="hyperion")
    model.add_disk(mass=M_disk, rmin=R_in, rmax=R_disk, plrho=alpha, h0=h_0, \
            plh=beta, dust=ddust)
    model.add_envelope(mass=M_env, rmin=R_in, rmax=R_env, cavpl=ksi, \
            cavrfact=f_cav, dust=edust)
    model.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    model.run_thermal(code="hyperion", nphot=2e6, mrw=True, pda=True, \
            mpi=True, nprocesses=nprocesses)
