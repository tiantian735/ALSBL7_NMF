from sklearn.decomposition import NMF
import numpy as np


#scikit run

def scikitRun(V,component,initialize,betaloss,tolerance,iteration):
    V = V.reshape(V.shape[0],(V.shape[1]*V.shape[2]))

    if np.any(V<0):
        V[V<0]=0
    nmf = NMF(n_components=component,init=initialize,beta_loss=betaloss,tol=tolerance,max_iter=iteration)
    W = nmf.fit_transform(V)
    H = nmf.components_  
    return W, H