#!/software/et_env/bin python

import numpy as np
import george
from george import kernels
import schwimmbad
import dynesty.plotting as dyplot
from dynesty import utils as dyfunc
import dynesty
import time

np.random.seed(seed=2568)
a=np.arange(0,3962,1)
np.random.shuffle(a)
a1,a2=np.split(a,2)

weights=np.load("./etgrid/3962_weights.npy")
coords=np.load("./etgrid/3962_coords.npy")

'''
Tstar=coords[a1,0]
logM_disk=coords[a1,2]
logR_disk=coords[a1,3]
h_0=coords[a1,4]
logR_in=coords[a1,5]
w2=weights[2][a1]
yerr=np.array([j*0.01 for j in w2])

x=np.transpose([Tstar,logM_disk,logR_disk,h_0,logR_in])
kernel = np.var(w2) \
    * kernels.ExpSquaredKernel(5000**2,ndim=5,axes=0) \
    * kernels.ExpSquaredKernel(3**2,ndim=5,axes=1) \
    * kernels.ExpSquaredKernel(1**2,ndim=5,axes=2) \
    * kernels.ExpSquaredKernel(0.1**2,ndim=5,axes=3) \
    * kernels.ExpSquaredKernel(0.5**2,ndim=5,axes=4) 


gp = george.GP(kernel)
gp.compute(x,yerr)

ndim = 6

def loglike (p):
    try:        
        gp.set_parameter_vector(p)
        return gp.log_likelihood(w2)
    except:
        return - 1e14

def ptform(u):
    #  [-50,50] [0,30] [-3,3.6] [-8,2.2] [-16,-1.4] [-10,2.5]
    return [u[0]*100 -50, u[1]*30 , u[2]*6.6-3 , \
            u[3]*10.2-8 , u[4]*14.6-16 , u[5]*12.5-10 ]


sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=1000, pool= schwimmbad.MultiPool())

print("live points finished")

t0=time.time()
sampler.run_nested()
print(time.time()-t0)
np.save("./dy_dropout/time_1.npy",time.time()-t0)
results = sampler.results

samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
new_samples = dyfunc.resample_equal(samples, weights)

np.save("./dy_dropout/dynesty_samples_1.npy",samples)
np.save("./dy_dropout/dynesty_newsamples_1.npy",new_samples)


# Plot a summary of the run.
rfig, raxes = dyplot.runplot(results)
# Plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(results)
# Plot the 2-D marginalized posteriors.
cfig, caxes = dyplot.cornerplot(results)

rfig.savefig("./dy_dropout/dyplot1_1.png")
tfig.savefig("./dy_dropout/dyplot2_1.png")
cfig.savefig("./dy_dropout/dyplot3_1.png")
'''


Tstar=coords[a2,0]
logM_disk=coords[a2,2]
logR_disk=coords[a2,3]
h_0=coords[a2,4]
logR_in=coords[a2,5]
w2=weights[2][a2]
yerr=np.array([j*0.01 for j in w2])

x=np.transpose([Tstar,logM_disk,logR_disk,h_0,logR_in])
kernel = np.var(w2) \
    * kernels.ExpSquaredKernel(5000**2,ndim=5,axes=0) \
    * kernels.ExpSquaredKernel(3**2,ndim=5,axes=1) \
    * kernels.ExpSquaredKernel(1**2,ndim=5,axes=2) \
    * kernels.ExpSquaredKernel(0.1**2,ndim=5,axes=3) \
    * kernels.ExpSquaredKernel(0.5**2,ndim=5,axes=4) 


gp = george.GP(kernel)
gp.compute(x,yerr)

ndim = 6

def loglike (p):
    try:        
        gp.set_parameter_vector(p)
        return gp.log_likelihood(w2)
    except:
        return - 1e14

def ptform(u):
    #  [-50,50] [0,30] [-3,3.6] [-8,2.2] [-16,-1.4] [-10,2.5]
    return [u[0]*100 -50, u[1]*30 , u[2]*6.6-3 , \
            u[3]*10.2-8 , u[4]*14.6-16 , u[5]*12.5-10 ]


sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=1000, pool= schwimmbad.MultiPool())

print("live points finished")

t0=time.time()
sampler.run_nested()
print(time.time()-t0)
np.save("./dy_dropout/time_2.npy",time.time()-t0)
results = sampler.results

samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
new_samples = dyfunc.resample_equal(samples, weights)

np.save("./dy_dropout/dynesty_samples_2.npy",samples)
np.save("./dy_dropout/dynesty_newsamples_2.npy",new_samples)


# Plot a summary of the run.
rfig, raxes = dyplot.runplot(results)
# Plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(results)
# Plot the 2-D marginalized posteriors.
cfig, caxes = dyplot.cornerplot(results)

rfig.savefig("./dy_dropout/dyplot1_2.png")
tfig.savefig("./dy_dropout/dyplot2_2.png")
cfig.savefig("./dy_dropout/dyplot3_2.png")


