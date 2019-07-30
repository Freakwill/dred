# dred
dred = dimension reducing (DR) for ml (suit to sklearn)

## Framework

X -DR-> X' -Regression-> Y' <-DR-> Y

## Usage

### Basic usage
```python
# X -> R^p, Y -> R^q
@DimReduce(p, q)
class cls(RegressorMixin):
    # Definition of cls, in sklearn form
```

### Create yourself dim reduction method

Make sure it has `transform` and `inverse_transform` method. :caution:

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
        self.dr1 = SVDTransformer(p)
        self.dr2 = SVDTransformer(q)


class PCADimReduce(DimReduce):
    # PCA for X and y
    def __init__(self, p=3, q=None):
        self.dr1 = PCA(n_components=p)
        self.dr2 = PCA(n_components=q)

```

## Example
see `solver.py`
run `lineqx.py`

## TODO
- [ ] More methods for DR
