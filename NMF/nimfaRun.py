import numpy as np
import nimfa








#nimfa run



def nimfaRun(V,rank,nrun,update,objective,maxiter):
    """
    Perform standard NMF factorization from nimfa package. 

    Return basis and mixture matrices of the fitted factorization model. 

    """

    V = V.reshape(V.shape[0],(V.shape[1]*V.shape[2]))
    
    
    if np.any(V<0):
        V[V<0]=0
    nmf = nimfa.Nmf(V, seed="random_c", rank=rank, n_run=nrun, update=update, objective=objective, max_iter=maxiter)
    fit = nmf()

    W = fit.basis()
    H = fit.coef()
    return W, H