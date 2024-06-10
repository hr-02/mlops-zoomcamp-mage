from typing import List, Callable, Dict, Tuple, Union

from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features

from mlops.utils.models.sklearn import load_class, train_model

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer



@transformer
def transform(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs
) -> Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    BaseEstimator,
]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    X, _, _ = vectorize_features(select_features(df))
    y: Series = df[target]

    X_train, X_val, dv = vectorize_features(
        select_features(df_train),
        select_features(df_val),
    )
    y_train = df_train[target]
    y_val = df_val[target]

    lr = LinearRegression()
    lr.fit(X, y)
    print(f" Y-intercept: {lr.intercept_}")
    
    return X, X_train, X_val, y, y_train, y_val, dv, lr








