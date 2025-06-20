import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as r_squared
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS

def simplify_teos(df):
    """Convert 'datetime64[ns, UTC]' to 'datetime64[ns, UTC]'"""
    df.index = df.index.tz_localize(None).normalize()
    return df

def log_returns(df):
    """Calculate log returns from price data"""
    df = np.log(df)
    df = df - df.shift(1)
    return df

def p_by_slice(X, y, t_list, t_list_labels):
    """Feature significance analysis by time slices"""
    feat_stats = pd.DataFrame(index=X.columns)

    for n, idx in enumerate(t_list):
        X_fit = X.loc[idx,:].dropna()
        y_fit = y.reindex(X_fit.index)
        feat_stats.loc[:,t_list_labels[n]] = r_regression(X_fit, y_fit, center=True)

    print('from', X_fit.index.min(), 'to', X_fit.index.max())
    return feat_stats

def p_by_year(X, y, sort_by='p_value', t_list=None):
    """Annual feature analysis"""
    feat_stats = pd.DataFrame(index=X.columns)

    for year in X.index.year.unique():
        X_fit = X.loc[str(year),:].dropna()
        y_fit = y.reindex(X_fit.index)
        feat_stats.loc[:,str(year)] = r_regression(X_fit, y_fit, center=True)

    print('from', X_fit.index.min(), 'to', X_fit.index.max())
    return feat_stats

def feature_profiles(X, y, sort_by='pearson', t_slice=None):
    """Comprehensive feature analysis"""
    if not t_slice:
        t_slice = slice(X.index.min(), X.index.max())
        print(t_slice)

    if t_slice is not None:
        X_fit = X.loc[t_slice,:].dropna().ravel()
        y_fit = y.reindex(X_fit.index).ravel()
    else:
        X_fit = X.ravel()
        y_fit = y.ravel()

    pear_test = r_regression(X_fit, y_fit, center=True)
    abs_pear_test = np.abs(pear_test)
    f_test, p_value = f_regression(X_fit, y_fit, center=False)
    t_test = np.sqrt(f_test)/np.sqrt(1)
    mi = mutual_info_regression(X_fit, y_fit)
    nobs = X_fit.count().unique()[0]
    feat_stats = pd.DataFrame({
        'nobs': nobs,
        'mutual_info': mi,
        'p_value': p_value,
        't_test': t_test,
        'pearson': pear_test,
        'abs_pearson': abs_pear_test
    }, index=X.columns)
    
    print('from', X_fit.index.min(), 'to', X_fit.index.max())
    return feat_stats.sort_values(by=[sort_by], ascending=False)

def generate_train_predict_calender(df, window_type=None, window_size=None):
    """Generate training and prediction date ranges"""
    date_ranges = []
    index = df.index
    num_days = len(index)

    if window_type == 'fixed':
        for i in range(0, num_days - window_size):
            train_start_date = index[i]
            train_end_date = index[i + window_size - 1]
            prediction_date = index[i + window_size]
            date_ranges.append([train_start_date, train_end_date, prediction_date])

    if window_type == 'expanding':
        for i in range(0, num_days - window_size):
            train_start_date = index[0]
            train_end_date = index[i + window_size - 1]
            prediction_date = index[i + window_size]
            date_ranges.append([train_start_date, train_end_date, prediction_date])

    return date_ranges

def graph_df(df, w=10, h=15):
    """Plot multiple time series in a DataFrame"""
    fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(w, h))

    for i, col in enumerate(df.columns):
        axes[i].plot(df[col])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()

class StatsModelsWrapper_with_OLS(BaseEstimator, RegressorMixin):
    """Wrapper for statsmodels OLS to work with sklearn"""
    def __init__(self, exog, endog):
        self.exog = exog
        self.endog = endog

    def fit(self, X_fit, y_fit):
        X_with_const = add_constant(X_fit, has_constant='add')
        self.model_ = OLS(y_fit, X_with_const)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X_pred):
        X_pred_constant = add_constant(X_pred, has_constant='add')
        return self.results_.predict(X_pred_constant)

    def summary(self, title):
        if title is not None:
            return self.results_.summary(title=title)
        else:
            return self.results_.summary(title="OLS Estimation Summary")

class EWMTransformer(BaseEstimator, TransformerMixin):
    """Exponential Weighted Moving Average transformer"""
    def __init__(self, halflife=3):
        self.halflife = halflife

    def fit(self, X, y=None):
        return self

    def set_output(self, as_dataframe=True):
        self.output_as_dataframe = as_dataframe
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_transformed = X.ewm(halflife=self.halflife).mean()
        if not self.output_as_dataframe:
            return X_transformed.values
        return X_transformed 