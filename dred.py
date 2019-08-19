#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DR method for regression in sklearn

dr1: DR for x data
dr2: DR for y data
"""

import types
import numpy.linalg as LA

def dredxy(m, dr1, dr2):
    # dred for fit method, or other methods defined as f(self, X, y)
    def mm(obj, X, y):
        dr1.fit(X)
        X = dr1.transform(X)
        if dr2:
            dr2.fit(y)
            y = dr2.transform(y)
        return m(obj, X, y)
    return mm

def dredx(m, dr1):
    # == dredxy(m, dr1, None)
    def mm(obj, X, y):
        dr1.fit(X)
        X = dr1.transform(X)
        return m(obj, X, y)
    return mm

def dredxy_(m, dr1, dr2):
    # have called dr1.fit, dr2.fit before calling dredxy_
    def mm(obj, X):
        X = dr1.transform(X)
        if dr2:
            return dr2.inverse_transform(m(obj, X))
        else:
            return m(obj, X)
    return mm

def dredx_(m, dr1):
    # == dredxy_(m, dr1, dr2)
    def mm(obj, X):
        X = dr1.transform(X)
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
            return Cp, Vp, s[:p]
        if self.p:
            X, V, s = svd(X, self.p)
            self.func = lambda X: X @ V
            self.inverse_func = lambda X: X @ V.T
            self.sigma = s.cumsum()
        return self


class DimReduce:
    """Decorator for dimension reduce

    Usage:
    @DimReduce(dr1, dr2)
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
        self.__dr1 = dr1
        self.__dr2 = dr2

    @property
    def dr1(self):
        return self.__dr1

    @property
    def dr2(self):
        return self.__dr2

    @dr1.setter
    def dr1(self, v):
        self.__dr1 = v

    @dr2.setter
    def dr2(self, v):
        self.__dr2 = v

    def __call__(self, cls):
        if self.dr2:
            cls.fit = types.MethodType(dredxy(cls.fit, self.dr1, self.dr2), cls)
            for m in ('transform', 'predict'):
                if hasattr(cls, m):
                    setattr(cls, m, types.MethodType(dredxy_(getattr(cls, m), self.dr1, self.dr2), cls))
        else:
            cls.fit = types.MethodType(dredx(cls.fit, self.dr1), cls)
            for m in ('transform', 'predict'):
                if hasattr(cls, m):
                    setattr(cls, m, types.MethodType(dredx_(getattr(cls, m), self.dr1), cls))
        def f(obj, k):
            if k == 'X':
                return self.dr1
            elif k == 'Y':
                return self.dr2
            else:
                raise KeyError(f'no such key {k}')
        cls.__getitem__ = types.MethodType(f, cls)
        return cls


class SVDDimReduce(DimReduce):
    # SVD for X and y
    def __init__(self, p=3, q=None):
        dr1 = SVDTransformer(p)
        dr2 = SVDTransformer(q)
        super(SVDDimReduce, self).__init__(dr1, dr2)


class PCADimReduce(DimReduce):
    # PCA for X and y
    def __init__(self, p=3, q=None):
        dr1 = PCA(n_components=p)
        dr2 = PCA(n_components=q)
        super(PCADimReduce, self).__init__(dr1, dr2)
