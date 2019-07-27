#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.linalg as LA

from utils import *
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split

import dred
def solver(p=30, q=3):
    @dred.SVDDimReduce(p, q)
    class cls(RegressorMixin):
        '''Linear equations
        XP = y
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

        def perf(self, n=10, *args, **kwargs):
            """Check the performance by running it several times
            
            Arguments:
                n {int} -- running times
            
            Returns:
                number -- mean time
            """
            import time
            times = []
            for _ in range(n):
                time1 = time.perf_counter()
                self.solve(*args, **kwargs)
                time2 = time.perf_counter()
                times.append(time2 - time1)
            return np.mean(times)
    return cls()


if __name__ == '__main__':


    from data import *
    A, A_test, B, B_test = train_test_split(A, B, test_size=0.2)
    lm = solver()
    lm.fit(A, B)
    B_ = lm.predict(A_test)
    print(lm.relerror(A_test, B_test))

