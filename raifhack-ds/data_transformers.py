import typing

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.exceptions import NotFittedError

class SmoothedTargetEncoding(BaseEstimator,TransformerMixin):
    """Регуляризованный таргет энкодинг.

    :param categorical_features: список из столбцов с категориальными признаками, которые нужно заэнкодить
    :param alpha: параметр регуляризации
    """

    def __init__(self, categorical_features: typing.List[str], alpha: float = 50.0):
        self.__is_fitted = False
        self.categorical_features = categorical_features
        self.alpha = alpha
        self.mean_price = None
        self.mean_price_by_cat = {}
        self.encoded_preffix = "encoded_"
        self.target = 'target'

    def smoothed_target_encoding(self, y: pd.Series) -> pd.Series:
        """Реализация регуляризованного таргед энкодинга.

        Принцип такой - чем меньше исходных данных, тем сильнее будет регуляризация
        Параметр регуляризации регуляризует мин. кол-во необходимых данных
        :param y: pd.Series с ценой
        :return: pd.Series с регуляризованной ценой
        """
        nrows = y.notnull().sum()
        return (y.mean() * nrows + self.alpha * self.mean_price) / (nrows + self.alpha)

    def fit(self, X: pd.DataFrame, y: typing.Union[np.array, pd.Series] = None):
        """На основе обучающей выборки запоминает средние цены в разрезе категорий.

        :param X: pd.DataFrame, обучающая выборка
        :param y: target
        :return:
        """
        X[self.target] = y
        self.mean_price = X[self.target].mean()
        for col in self.categorical_features:
            self.mean_price_by_cat[col] = (
                X.groupby(col)[self.target].apply(lambda x: self.smoothed_target_encoding(x)).fillna(self.mean_price)
            )

        X.drop(self.target, axis=1, inplace=True)
        self.__is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: typing.Union[np.array, pd.Series] = None):
        """Применение регуляризованного таргет энкодинга.

        :param X: pd.DataFrame, обучающая выборка
        :return:
        """
        X_cp = X.copy()
        if self.__is_fitted:
            encoded_cols = []
            for col in self.categorical_features:
                new_col = self.encoded_preffix + col
                X_cp[new_col] = X_cp[col].map(self.mean_price_by_cat[col]).fillna(self.mean_price)
                encoded_cols.append(new_col)
            return X_cp[encoded_cols]
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet. Call 'fit' with appropriate arguments before using this transformer".format(
                    type(self).__name__
                )
            )