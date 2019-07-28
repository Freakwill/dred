#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import types
import numpy.linalg as LA

def decoxy(m, dr1, dr2):
    def mm(obj, X, y):
        dr1.fit(X)
        X = dr1.transform(X)
        if dr2:
            dr2.fit(y)
            y = dr2.transform(y)
        return m(obj, X, y)
    return mm

def decox(m, dr1, dr2):
    def mm(obj, X):
        X = dr1.transform(X)
        if dr2:
            return dr2.inverse_transform(m(obj, X))
        else:
            return m(obj, X)
    return mm


from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

class SVDTransformer(FunctionTransformer):
    '''SVD DR transformer
    '''
    def __init__(self, p=None, *args, **kwargs):
        super(SVDTransformer, self).__init__(*args, **kwargs)
        self.p = p


    def fit(self, X):
        def svd(X, p):
            V, s, Vh = LA.svd(X.T @ X)
            Vp = V[:, :p]
            Cp = X @ Vp
            return Cp, Vp
        if self.p:
            X, V = svd(X, self.p)
            self.func = lambda X: X @ V
            self.inverse_func = lambda X: X @ V.T


class DimReduce:
    """Decorator for dimension reduce

    Usage:
    @DimReduce(p, q)
    class cls(RegressorMixin):
        Definition of cls, in sklearn form
    
    Example:
    @SVDDimReduce(p, q)
    class cls(RegressorMixin):
        '''Linear Regressor
        X P = y
        '''

        def fit(self, X, y):
            self.P = LA.lstsq(X, y, rcond=None)[0]

        def error(self, X, y):
            return error(self.predict(X), y)

        def relerror(self, X, y):
            return relerror(self.predict(X), y)

        def transform(self, X):
            return X @ self.P

        def predict(self, X):
            return X @ self.P

    """

    def __init__(self, dr1, dr2=None):
        self.dr1 = dr1
        self.dr2 = dr2

    def __call__(self, cls):
        cls.fit = types.MethodType(decoxy(cls.fit, self.dr1, self.dr2), cls)
        for m in ('transform', 'predict'):
            setattr(cls, m, types.MethodType(decox(getattr(cls, m), self.dr1, self.dr2), cls))
        return cls


class SVDDimReduce(DimReduce):
    # SVD for X and y
    def __init__(self, p=3, q=None):
        self.dr1 = SVDTransformer(p)
        self.dr2 = SVDTransformer(q)


class PCADimReduce(DimReduce):
    # PCA for X and y
    def __init__(self, p=3, q=None):
        self.dr1 = PCA(n_components=p)
        self.dr2 = PCA(n_components=q)
