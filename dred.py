#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import types
import numpy.linalg as LA

def decoxy(m, dr1, args1, dr2, args2):
    def mm(obj, X, y):
        X, V = dr1(X, *args1)
        obj.Xdr = X
        obj.rm1 = V
        if dr2:
            y, W = dr2(y, *args2)
            obj.ydr = y
            obj.rm2 = W
            return m(obj, X, y)
        else:
            return m(obj, X, y)
    return mm

def decox(m, flag=False):
    def mm(obj, X):
        if flag:
            X = obj.Xdr
        else:
            X = X @ obj.rm1
        if obj.rm2 is not None:
            return m(obj, X) @ obj.rm2.T
        else:
            return m(obj, X)
    return mm


class DimReduce:
    """Decorator for dimension reduce

    Usage:
    @SVDDimReduce(p, q)
    class cls(egressorMixin):
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

    def __init__(self, dr1, args1, dr2=None, args2=None):
        self.dr1 = dr1
        self.args1 = args1
        self.dr2 = dr2
        self.args2 = args2
        self.rm1 = self.rm2 = None
        self.Xdr = None

    def __call__(self, cls):
        cls.fit = types.MethodType(decoxy(cls.fit, self.dr1, self.args1, self.dr2, self.args2), cls)
        for m in ('transform', 'predict'):
            setattr(cls, m, types.MethodType(decox(getattr(cls, m)), cls))
        return cls


class SVDDimReduce(DimReduce):
    # SVD for X and y
    def __init__(self, p=3, q=None):
        def svd(X, p):
            V, s, Vh = LA.svd(X.T @ X)
            Vp = V[:, :p]
            Cp = X @ Vp
            return Cp, Vp
        self.dr1 = svd
        self.dr2 = svd
        self.args1 = (p,)
        self.args2 = (q,)
