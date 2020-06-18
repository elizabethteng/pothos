import numpy as np
import george
from george import kernels
from scipy.optimize import minimize
from time import time
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--rname", help="name of the pca weights read in for training data",type=str)
parser.add_argument("--wname", help="name of the results being written out, minus filetype",type=str)
parser.add_argument("--verbose",help="print out chisq and hyperparameter vector in the likelihood function",action="store_true")
parser.add_argument("--solver",help="scipy optimize minimize solver", type=str)
parser.add_argument("--error",help="percent error for GP computation",type=int)
args=parser.parse_args()

coords=np.load("./etgrid/"+args.rname+"_coords.npy")
eigenseds=np.load("./etgrid/"+args.rname+"_eigenseds.npy")
weights=np.load("./etgrid/"+args.rname+"_weights.npy")
pcamean=np.load("./etgrid/"+args.rname+"_mean.npy")
sedsflat=np.load("./etgrid/sedsflat.npy")

yerrs=[]
for i in range(16):
    yerrs.append([x*0.01*args.error for x in weights[i]])
    
initvecs=[]
for i in range(16):
    initvecs.append([ 6.33043185, 18.42068074,  0.,  0., -0.4462871 , -5.05145729, -1.38629436, 
                     -2.4079456086518722, -2.4079456086518722,  -3.2188758248682006, -3.21887582, 
                     -2.77258872, -2.77258872,  1.38629436,  2.19722458, 0.8109302162163288])

kernel = 16*kernels.ExpSquaredKernel(15**2,ndim=15,axes=0)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=1)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=2)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=3)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=4)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=5)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=6)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=7)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=8)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=9)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=10)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=11)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=12)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=13)*\
        kernels.ExpSquaredKernel(1**2,ndim=15,axes=14) 
blankhodlr=george.GP(kernel,solver=george.HODLRSolver)

def F_chisq_quiet(hp,gp):
    if args.solver=="COBYLA":
        hyperparams=np.transpose(np.array(hp).reshape(16,16))
    else:
        hyperparams=np.array(hp).reshape(16,16)
    preds=[]
    for i in range(len(weights)):  # same covfunc for each weight and the sample mean
        gp.set_parameter_vector(hyperparams[i])
        if args.verbose==True:
            print("weight #"+str(i))
            print(gp.get_parameter_vector())
            conv.append(gp.get_parameter_vector())
        gp.compute(coords,yerrs[i])
        pred, pred_var = gp.predict(weights[i], coords, return_var=True)
        preds.append(pred)
    reconst_SEDs=[]
    for i in range(len(coords)):
        reconst=np.dot(np.array(preds)[:,i][0:15],eigenseds[0:15]) + pcamean + np.array(preds)[:,i][15]
        reconst_SEDs.append(reconst)
    allsedsflat=np.ndarray.flatten(np.array(reconst_SEDs))
    chisq=np.sum((sedsflat-allsedsflat)**2/0.1)
    if args.verbose==True:
        print(chisq)
    return chisq

def chisq(p):
    return F_chisq_quiet(p,blankhodlr)


print("starting minimize routine")
conv=[]
t0=time()
result = minimize(chisq,initvecs,method=args.solver)
print("minimize routine done in %0.3fs" % (time() - t0))

print("Final chisq: ")
print(np.array(result.x).reshape(16,16))

np.save("./etgrid/"+args.wname+"_optimize_result.npy",np.array(result.x).reshape(16,16))
np.save("./etgrid/"+args.wname+"_convergence.npy",np.array(conv))
np.save("./etgrid/"+args.wname+"_time_rec.npy",(time() - t0))









