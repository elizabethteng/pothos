#!/software/et_env/bin python

import numpy as np
import george
from george import kernels
import schwimmbad
import dynesty.plotting as dyplot
from dynesty import utils as dyfunc
import dynesty
import time

weights=np.load("./etgrid/3976_weights.npy")
coords=np.load("./etgrid/3976_coords.npy")
Tstar=coords[:,0]
logL_star=coords[:,1]
logM_disk=coords[:,2]
logR_disk=coords[:,3]
h_0=coords[:,4]
logR_in=coords[:,5]
gamma_t=coords[:,6]
bix=coords[:,7]
logM_env_t = coords[:,8]
logR_env = coords[:,9]
f_cav=coords[:,10]
ksi = coords[:,11]
loga_max=coords[:,12]
p=coords[:,13]
biy=coords[:,14]
w2=weights[0]
yerr=np.array([j*0.01 for j in w2])

x=np.transpose([Tstar, logL_star,logM_disk,logM_env_t,logR_env])
kernel = np.var(w2) \
    * kernels.ExpSquaredKernel(20000**2,ndim=5,axes=0) \
    * kernels.ExpSquaredKernel(0.35**2,ndim=5,axes=1) \
    * kernels.ExpSquaredKernel(0.47**2,ndim=5,axes=2) \
    * kernels.ExpSquaredKernel(0.1**2,ndim=5,axes=3) \
    * kernels.ExpSquaredKernel(0.2**2,ndim=5,axes=4) 

gp = george.GP(kernel)
gp.compute(x,yerr)

ndim = 6

def loglike(p):
    try:        
        gp.set_parameter_vector(p)
        return gp.log_likelihood(w2)
    except:
        return - 1e14

def ptform(u):
    #  [-50,50] [0,30] [-5, 5] [-6.6,3] [-8,0] [-4,0.3]
    return [u[0]*100 -50, u[1]*30 , u[2]*10-5 , \
            u[3]*9.6-6.6 , u[4]*8-8 , u[5]*4.3-4 ]

print("testing live points")

sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=1000, pool= schwimmbad.MultiPool())

print("live points finished")

t0=time.time()
sampler.run_nested()
print(time.time()-t0)
np.save("./dy_01289_w0_v2/dynesty_timerec.npy",time.time()-t0)
results = sampler.results

samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
new_samples = dyfunc.resample_equal(samples, weights)

np.save("./dy_01289_w0_v2/dynesty_samples.npy",samples)
np.save("./dy_01289_w0_v2/dynesty_newsamples.npy",new_samples)

rfig, raxes = dyplot.runplot(results)
tfig, taxes = dyplot.traceplot(results)
cfig, caxes = dyplot.cornerplot(results)

rfig.savefig("./dy_01289_w0_v2/dyplot1.png")
tfig.savefig("./dy_01289_w0_v2/dyplot2.png")
cfig.savefig("./dy_01289_w0_v2/dyplot3.png")

