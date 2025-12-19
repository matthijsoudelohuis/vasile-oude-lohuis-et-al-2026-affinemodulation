"""
@author: Matthijs oude Lohuis
Champalimaud 2023

"""

import scipy as sp
import numpy as np
from scipy import linalg
from scipy.linalg import orth, qr, svd
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.decomposition import PCA, FactorAnalysis,FastICA
from sklearn.model_selection import KFold
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils.psth import construct_behav_matrix_ts_F

def EV(Y,Y_hat):
    # e = Y - Y_hat
    # ev = 1 - np.trace(e.T @ e) / np.trace(Y.T @ Y) #fraction of variance explained
    ev = 1 - np.nanvar(Y-Y_hat) / np.nanvar(Y)
    return ev

def LM(Y, X, lam=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    # Y_hat = X @ B_hat
    return B_hat

def Rss(Y, Y_hat, normed=True):
    """ evaluate (normalized) model error """
    e = Y_hat - Y
    Rss = np.trace(e.T @ e)
    if normed:
        Rss /= Y.shape[0]
    return Rss

def low_rank_approx(A, r, mode='left'):
    """ calculate low rank approximation of matrix A and 
    decomposition into L and W """
    # decomposing and low rank approximation of A
    U, s, Vh = linalg.svd(A)
    S = linalg.diagsvd(s,U.shape[0],s.shape[0])

    return U[:,:r],S[:r,:r],Vh[:r,:]


def RRR(Y, X, B_hat, r):
    """ reduced rank regression by low rank approx of Y_hat """
    
    Y_hat = X @ B_hat

    U,S,V = low_rank_approx(Y_hat,r)

    # Y_hat_rr =  X @ B_hat @ U @ U.T
    Y_hat_rr =  U @ U.T @ X @ B_hat

    return Y_hat_rr,U,S,V

def RRR_cvR2(Y, X, rank,lam=0,kfold=5):
    # Input: 
    # Y is activity in area 2, X is activity in area 1

    # Function:
    # X is of shape K x N (samples by features), Y is of shape K x M
    # K is the number of samples, N is the number of neurons in area 1,
    # M is the number of neurons in area 2

    # multiple linear regression, B_hat is of shape N x M:
    # B_hat               = LM(Y,X, lam=lam) 
    #RRR: do SVD decomp of Y_hat, 
    # U is of shape K x r, S is of shape r x r, V is of shape r x M
    # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r)

    kf      = KFold(n_splits=kfold,shuffle=True)

    R2_cv_folds = np.full((kfold),np.nan)

    X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
    Y                   = zscore(Y,axis=0)

    for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
        
        X_train, X_test     = X[idx_train], X[idx_test]
        Y_train, Y_test     = Y[idx_train], Y[idx_test]

        B_hat_train         = LM(Y_train,X_train, lam=lam)

        Y_hat_train         = X_train @ B_hat_train

        # decomposing and low rank approximation of A
        # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
        
        U, s, V = svds(Y_hat_train,k=rank,which='LM')

        B_rrr               = B_hat_train @ V.T @ V #project beta coeff into low rank subspace

        Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace

        R2_cv_folds[ikf] = EV(Y_test,Y_hat_rr_test)

    return np.nanmean(R2_cv_folds)

def regress_out_behavior_modulation(ses,X=None,Y=None,nvideoPCs = 30,rank=None,nranks=None,lam=0,perCond=False,kfold = 5):
    
    if X is None:
        idx_T   = np.ones(len(ses.trialdata),dtype=bool)
        X       = np.stack((ses.respmat_videome[idx_T],
                        ses.respmat_runspeed[idx_T],
                        ses.respmat_pupilarea[idx_T],
                        ses.respmat_pupilx[idx_T],
                        ses.respmat_pupily[idx_T]),axis=1)
        X       = np.column_stack((X,ses.respmat_videopc[:nvideoPCs,idx_T].T))
        X       = zscore(X,axis=0,nan_policy='omit')

        si      = SimpleImputer()
        X       = si.fit_transform(X)

        # X,Xlabels = construct_behav_matrix_ts_F(ses,nvideoPCs=nvideoPCs)

    if Y is None:
        # Y = ses.calciumdata.to_numpy()

        Y               = ses.respmat[:,idx_T].T
        Y               = zscore(Y,axis=0,nan_policy='omit')

    assert X.shape[0] == Y.shape[0],'number of samples of calcium activity and interpolated behavior data do not match'

    if rank is None:
        if nranks is None: 
            nranks = X.shape[1]
        
        R2_cv_folds = np.full((nranks,kfold),np.nan)
        kf = KFold(n_splits=kfold,shuffle=True)
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
            X_train, X_test     = X[idx_train], X[idx_test]
            Y_train, Y_test     = Y[idx_train], Y[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=lam)

            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of A
            # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
            U, s, V = svds(Y_hat,k=nranks,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
    
            S = linalg.diagsvd(s,U.shape[0],s.shape[0])
            for r in range(nranks):
                B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace
                R2_cv_folds[r,ikf] = EV(Y_test,Y_hat_rr_test)

        repmean,rank = rank_from_R2(R2_cv_folds,nranks,kfold)

    if perCond:
        Y_hat_rr = np.zeros(Y.shape)
        conds = np.unique(ses.trialdata['stimCond'])
        for ic in conds:
            idx_c = ses.trialdata['stimCond']==ic

            B_hat           = LM(Y[idx_c,:],X[idx_c,:],lam=lam)

            Y_hat           = X[idx_c,:] @ B_hat

            # decomposing and low rank approximation of Y_hat
            # U, s, V = linalg.svd(Y_hat)
            U, s, V = svds(Y_hat,k=rank)
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            S = linalg.diagsvd(s,U.shape[0],s.shape[0])

            #construct low rank subspace prediction
            Y_hat_rr[idx_c,:]       = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]

        Y_out           = Y - Y_hat_rr #subtract prediction
    else:

        B_hat           = LM(Y,X,lam=lam)

        Y_hat           = X @ B_hat

        # decomposing and low rank approximation of Y_hat
        # U, s, V = linalg.svd(Y_hat,full_matrices=False)
        U, s, V = svds(Y_hat,k=rank)
        U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

        S = linalg.diagsvd(s,U.shape[0],s.shape[0])

        #construct low rank subspace prediction
        Y_hat_rr       = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]

        Y_out           = Y - Y_hat_rr #subtract prediction

    # print("EV of behavioral modulation: %1.4f" % EV(Y,Y_hat_rr))

    return Y,Y_hat_rr,Y_out,rank,EV(Y,Y_hat_rr)

