#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Error Curve

solve the eq, then draw the error curve
"""


import numpy as np
import numpy.linalg as LA
from sklearn.model_selection import train_test_split

from utils import *
from data import *
from solver import *

if __name__ == '__main__':

    A, A_test, B, B_test = train_test_split(A, B, test_size=0.2)

    Es1 = []
    Es2 = []

    V, s, Vh = LA.svd(A.T @ A)

    s /= np.sum(s)
    ss = np.cumsum(s)
    ps = np.arange(1, 81, 5)

    for p in ps:
        s = solver(p)
        s.fit(A, B)

        E1 = s.relerror(A, B)
        E2 = s.relerror(A_test, B_test)

        Es1.append(E1)
        Es2.append(E2)
 
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

    ax = plt.subplot(111)
    ax.set_xlabel("Number of PCs of A", fontproperties=myfont)
    ax.set_ylabel("Relative error", fontproperties=myfont)

    #ax.set_title('Number of PCs-error', fontproperties=myfont)
    ax.plot(ps, Es1, '-o', ps, Es2, '-s')

    ax.legend(('Error after DR', 'Predict Error'), prop=myfont)

    XX = A @ LA.lstsq(A, B, rcond=None)[0] - B
    re = error(XX) / error(B)
    ax.plot((ps[0], ps[-1]), [re, re], '--k')
    ax.annotate('Error of original Equation', xy = (ps[0], re), xytext=(ps[0], re + 0.1), arrowprops={'arrowstyle':'->'}, fontproperties=myfont)

    ret = error(A_test @ LA.lstsq(A_test, B_test, rcond=None)[0], B_test) / error(B_test)
    ax.plot((ps[0], ps[-1]), [ret, ret], '--g')
    ax.annotate('Relative error of predict', color='green', xy=(ps[0], ret), xytext=(ps[0], ret - 0.1), arrowprops={'arrowstyle':'->', 'color':'green'}, fontproperties=myfont)

    tax = ax.twinx()
    tax.plot(ps, ss[:80:5], 'm-.')
    tax.set_ylabel('Accumulative contribution', fontproperties=myfont)
    tax.legend(('Accumulative contribution',), prop=myfont)
 
    plt.show()
