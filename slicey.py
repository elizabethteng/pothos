import numpy as np
import pickle
from time import time
import pdspy.modeling as modeling
import pdspy.dust as dust
import numpy as np

param_names = ["Tstar","logLstar","logMdisk","logRdisk","h0","logRin","gamma",\
               "bix","logMenv","logRenv","fcav","ksi","logamax","p","biy"]

ranges = [[3000.,5000.], [-1,3.],[-8.,-2.], [0.,3.],[0.01,0.5],[-1.,2.5],[-1.,np.log10(2.1)],\
        [0.,2.],[np.log10(0.5),np.log10(6.5)],[2.5,4.], [0.,1.], [0.5,1.5],[0.,5.],[2.5,4.5],[-1.25,0.75]]
steps=[]
bases=[]
for i in range(len(ranges)):
    steps.append(np.linspace(ranges[i][0],ranges[i][1],11))
    bases.append(steps[i][5])

def run_yso_model(filename,name,Tstar=None, logL_star=None, \
        logM_disk=None, logR_disk=None, h_0=None, logR_in=None, gamma=None, \
        beta=None, logM_env=None, logR_env=None, f_cav=None, ksi=None, \
        loga_max=None, p=None, incl=None):
    
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
    
    # Run the thermal simulation.
    model.run_thermal(code="radmc3d", nphot=1e6, \
            modified_random_walk=True, verbose=False, setthreads=20, \
            timelimit=10800*2.5)

    # Run the SED.

    model.set_camera_wavelength(np.logspace(-1.,4.,500))

    model.run_sed(name="SED", nphot=1e5, loadlambda=True, incl=incl, \
            pa=0., dpc=140., code="radmc3d", camera_scatsrc_allfreq=True, \
            verbose=False, setthreads=20)
    # run the image
    model.run_image(name="1mm", nphot=1e5, npix=512, \
            pixelsize=0.05, lam="1000", incl=0., pa=0.,\
            dpc=140, verbose=True)
        
    # Write out the file.
    model.write_yso("../../grid/slices/"+filename)
    
    print(name+" generated")

slicenames=[]
slicevals=[]
for i in range(len(param_names)):
    slicenames.append([])
    slicevals.append([])
    
def make_slice(paramindex,basevals):
    slicevals[paramindex]=[] # make sure lists are empty
    slicenames[paramindex]=[]
    parvals=basevals # set base values
    
    for i in range(11): # for each model in the slice:
        
        parvals[paramindex]=steps[paramindex][i] # set that parameter's value
        
        filename="" # find the filename
        for j in range(len(parvals)):
            filename+=param_names[j]+"_"+str(parvals[j])+"_"
        filename=filename[:-1]
        filename+=".hdf5"
               
        try:
            run_yso_model(filename,name="param "+str(paramindex)+" model "+str(i), Tstar=parvals[0], logL_star=parvals[1], \
                logM_disk=parvals[2], logR_disk=parvals[3], h_0=parvals[4], logR_in=parvals[5], gamma=parvals[6], \
                beta=parvals[7], logM_env=parvals[8], logR_env=parvals[9], f_cav=parvals[10], ksi=parvals[11], \
                loga_max=parvals[12], p=parvals[13], incl=parvals[14])
            model=modeling.YSOModel()
            model.read_yso("../grid/slices/"+filename)
            fluxvals=np.log10(model.spectra["SED"].flux)

            slicenames[paramindex].append(filename)
            slicevals[paramindex].append(fluxvals)
        except:
            print(paramindex, basevals, "failed")
for m in [8,14]:
        
    bases=[]
    for i in range(len(param_names)):
        steps.append(np.linspace(ranges[i][0],ranges[i][1],11))
        bases.append(steps[i][5])
        
    t0=time()
    make_slice(m,bases)
    print("slice for "+param_vals[m]+" done in %0.3fs" % (time() - t0))
    print(" ")     
    
np.save("../slicenames814.npy",slicenames)
np.save("../slicevals814.npy",slicevals)
        