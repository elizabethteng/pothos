#!/software/et_env/bin python

import numpy as np
import george
from george import kernels
import schwimmbad
import dynesty.plotting as dyplot
from dynesty import utils as dyfunc
import dynesty
import time
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument("--p", help="params for dynesty fit",nargs='+',type=int)
parser.add_argument("--w", help="weight", type=int)
parser.add_argument("--v", help="version", type=int)
pa=parser.parse_args().p

weights=np.load("./etgrid/3976_weights.npy")
coords=np.load("./etgrid/3976_coords.npy")

w2=weights[parser.parse_args().w]
yerr=np.array([j*0.01 for j in w2])

pr=np.load("./priors.npy")

x=[]
for i in pa:
    x.append(coords[:,i])
x=np.transpose(x)
kernel = np.var(w2) 
for i in range(len(pa)):
    kernel*= kernels.ExpSquaredKernel(1**2,ndim=len(pa),axes=i)

gp = george.GP(kernel)
gp.compute(x,yerr)

ndim = len(pa)+1

def loglike(p):
    try:        
        gp.set_parameter_vector(p)
        return gp.log_likelihood(w2)
    except:
        return - 1e14

def ptform(u):
    priors=[u[0]*15 -5]
    for i in range(len(pa)):
        priors.append(u[i+1]*pr[pa[i]][2]+pr[pa[i]][0])
    return priors

print("testing live points")

sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=1000, pool= schwimmbad.MultiPool())

print("live points finished")


fname="dy_"
for i in pa:
    fname+=str(i)
fname+="_w"+str(parser.parse_args().w)
fname+="_v"+str(parser.parse_args().v)

direc=os.path.join("./"+fname)
if not os.path.exists(direc):
    os.mkdir(direc)

t0=time.time()
sampler.run_nested()
print(time.time()-t0)
np.save("./"+fname+"/dynesty_timerec.npy",time.time()-t0)
results = sampler.results

samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
new_samples = dyfunc.resample_equal(samples, weights)

np.save("./"+fname+"/dynesty_samples.npy",samples)
np.save("./"+fname+"/dynesty_newsamples.npy",new_samples)

rfig, raxes = dyplot.runplot(results)
tfig, taxes = dyplot.traceplot(results)
cfig, caxes = dyplot.cornerplot(results)

rfig.savefig("./"+fname+"/dyplot1.png")
tfig.savefig("./"+fname+"/dyplot2.png")
cfig.savefig("./"+fname+"/dyplot3.png")

