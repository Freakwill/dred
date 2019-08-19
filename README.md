# dred
![](https://github.com/Freakwill/dred/blob/master/logo.png)

dred = dimension reducing (DR) for ML (suit to sklearn)

Currently, only for regression.

## Framework

`X -DR-> X' -Regression-> Y' <-DR-> Y`

train data X, Y
test(predict) data Xtest, Ytest
1. X and Y will be dred-ed
2. a model will fit X' and Y'
3. After Xtest is dred-ed, predict the output of Xtest'
4. Get Y', reconstruct Y from Y'
5. get error of Y to Ytest

## Requirements
scikit-learn

## Usage

### Basic usage
```python
# X -> R^p, Y -> R^q
@DimReduce(p, q)
class cls(RegressorMixin):
    # Definition of cls, in sklearn form
```

### Create yourself dim reduction method

I've defined two DR methods in the module

```python
class SVDTransformer(FunctionTransformer):
    '''SVD DR transformer
    
    make sure it has transform and inverse_transform method
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

```

Make sure dr1 and dr2 has `transform` and `inverse_transform` method. :caution:

## Example
see `solver.py`
run `lineqx.py` (demo program)

![](https://github.com/Freakwill/dred/blob/master/demo.png)


## TODO
- [ ] More methods for DR
- [ ] make Classifier
