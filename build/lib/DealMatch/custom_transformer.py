from sklearn.pipeline import TransformerMixin
import numpy as np


class DenseTransformer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X).__module__ == np.__name__:
            return X
        return X.toarray()
