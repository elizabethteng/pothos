#!/software/et_env/bin python

import numpy as np
import george
from george import kernels
import schwimmbad
import dynesty.plotting as dyplot
from dynesty import utils as dyfunc
import dynesty
import time

weights=np.load("./etgrid/3962_weights.npy")
coords=np.load("./etgrid/3962_coords.npy")
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
w2=weights[2]
yerr=np.array([j*0.01 for j in w2])

x=np.transpose([logL_star,logM_disk,bix,logM_env_t,biy])
kernel = np.var(w2) \
    * kernels.ExpSquaredKernel(5000**2,ndim=5,axes=0) \
    * kernels.ExpSquaredKernel(3**2,ndim=5,axes=1) \
    * kernels.ExpSquaredKernel(0.3**2,ndim=5,axes=2) \
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
        return - 1e10

def ptform(u):
    #  [-50,50] [-1.4,4.6] [-1.3,3.6] [-4.6,-1.4] [-6,0] [-4.6,-1.4]
    return [u[0]*100 -50, u[1]*6-1.4 , u[2]*4.9-1.3 , \
            u[3]*6-4.6 , u[4]*6-6 , u[5]*6-4.6 ]

sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=1000, pool= schwimmbad.MultiPool())

print("live points finished")

t0=time.time()
sampler.run_nested()
print(time.time()-t0)
np.save("./dy_127814/dynesty_timerec.npy",time.time()-t0)
results = sampler.results

samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
new_samples = dyfunc.resample_equal(samples, weights)

np.save("./dy_127814/dynesty_samples.npy",samples)
np.save("./dy_127814/dynesty_newsamples.npy",new_samples)
np.save("./dy_127814/dy_results_127814.npy",results)


# Plot a summary of the run.
rfig, raxes = dyplot.runplot(results)
# Plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(results)
# Plot the 2-D marginalized posteriors.
cfig, caxes = dyplot.cornerplot(results)

rfig.savefig("./dy_127814/dyplot1.png")
tfig.savefig("./dy_127814/dyplot2.png")
cfig.savefig("./dy_127814/dyplot3.png")

