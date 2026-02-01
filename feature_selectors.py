import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from boruta import BorutaPy
# =====================================================
# ElasticNet
# =====================================================
class ENetSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cv=10, random_state=42):
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):

        self.model_ = ElasticNetCV(
            cv=self.cv,
            random_state=self.random_state,
            max_iter=10000,
            n_jobs=-1
        )

        self.model_.fit(X, y)

        self.mask_ = np.abs(self.model_.coef_) > 0

        return self

    def transform(self, X):

        if self.mask_.sum() == 0:
            return X

        return X[:, self.mask_]


# =====================================================
# LASSO
# =====================================================
class LassoSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cv=10, random_state=42):
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):

        self.model_ = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            max_iter=5000
        )

        self.model_.fit(X, y)

        self.mask_ = self.model_.coef_ != 0

        return self

    def transform(self, X):

        if self.mask_.sum() == 0:
            return X

        return X[:, self.mask_]


# =====================================================
# Random Forest
# =====================================================
class RFSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state
        )

        self.model_.fit(X, y)

        imp = self.model_.feature_importances_

        self.mask_ = imp > 0

        return self

    def transform(self, X):

        if self.mask_.sum() == 0:
            return X

        return X[:, self.mask_]


# =====================================================
# Mutual Information
# =====================================================
class MISelector(BaseEstimator, TransformerMixin):

    def __init__(self, k=500):
        self.k = k

    def fit(self, X, y):

        scores = mutual_info_regression(X, y)

        idx = np.argsort(scores)[::-1][:self.k]

        self.mask_ = np.zeros(X.shape[1], dtype=bool)
        self.mask_[idx] = True

        return self

    def transform(self, X):

        return X[:, self.mask_]


# =====================================================
# XGBoost
# =====================================================
class XGBSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_estimators=100, random_state=42):

        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):

        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            objective="reg:squarederror",
            random_state=self.random_state
        )

        self.model_.fit(X, y)

        imp = self.model_.feature_importances_

        self.mask_ = imp > 0

        return self

    def transform(self, X):

        if self.mask_.sum() == 0:
            return X

        return X[:, self.mask_]


# =====================================================
# Boruta
# =====================================================
class BorutaSelector(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=42, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        rf = RandomForestRegressor(
            n_estimators=1000,
            n_jobs=-1,
            random_state=self.random_state
        )

        self.boruta_ = BorutaPy(
            rf,
            n_estimators="auto",
            random_state=self.random_state,
            verbose=self.verbose
        )

        self.boruta_.fit(X, y)

        self.mask_ = self.boruta_.support_

        return self

    def transform(self, X):

        if self.mask_.sum() == 0:
            return X

        return X[:, self.mask_]


# =====================================================
# LightGBM
# =====================================================
class LGBMSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_estimators=100, random_state=42):

        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):

        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )

        self.model_.fit(X, y)

        imp = self.model_.feature_importances_

        self.mask_ = imp > 1e-5

        return self

    def transform(self, X):

        if self.mask_.sum() == 0:
            return X

        return X[:, self.mask_]
