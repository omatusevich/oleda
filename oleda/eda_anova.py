"""Exploratory Data Analysis with python.

Automatic report generation from a pandas DataFrame.
Insights from data.

Typical usage example:

  oleda.pairwise_report(df1,df2)
"""
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as pls

import seaborn as sns
from IPython.display import display, HTML
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.formula.api import ols

warnings.filterwarnings('ignore')

#=====================#=====================#=====================#=====================
# target variance among categorical feature values
#=====================#=====================#=====================#=====================

def turkeyHSD(df,feature,target):
    # Tukey HSD test
    # target mean on feature values
    comp = mc.MultiComparison(df[target],df[feature])
    post_hoc_res = comp.tukeyhsd()
    results_as_html =post_hoc_res.summary().as_html()
    results_as_pandas = pd.read_html(results_as_html, header=0, index_col=0)[0]
    return (results_as_pandas[results_as_pandas.reject].sort_values('meandiff',ascending=False)
            if results_as_pandas.shape[0]>0 and
            results_as_pandas[results_as_pandas.reject].shape[0]>0
            else results_as_pandas)


def anova(df,feature,target,verbose=True):
    """Anova test with statsmodels.

    Peforms Shapiro-Wilk test to check normality first. If it passes do Anova test.

    Args:
        df  (DataFrame): pandas DataFrame
        feature (string): Categorical feature name.
        target  (string):  Numerical feature name .

    Returns:
        categories that passes anova test.

    """
    model = ols(target+' ~ C('+feature+')', data=df).fit()

    #Shapiro-Wilk test to check the normal distribution of residuals
    w, pvalue = stats.shapiro(model.resid)
    if verbose:
        print('\nShapiro-Wilk test - normal distribution of residuals . ', 'OK' if (pvalue <0.05) else 'KO' )
        print(w, pvalue)
    if pvalue>0.05:
        return False

    anova_table = sm.stats.anova_lm(model, typ=2)
    if verbose:
        print(anova_table)
        print('Anova ' ,anova_table['PR(>F)'].iloc[0], 'OK' if anova_table['PR(>F)'].iloc[0] <0.05 else 'KO')

    return anova_table['PR(>F)'].iloc[0]<0.05

#=====================#=====================#=====================#=====================
# two way anova
#=====================#=====================#=====================#=====================

def two_way_anova(df,feature1,feature2,target):
    """Two way anova with statsmodel."""
    model = ols(target+' ~ C('+feature1+') + C('+feature2+') + C('+feature1+'):C('+feature2+')', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def get_feature_type(s):
    if s.dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        return 'Numeric'
    elif '[ns]' in str(s.dtype) or 'datetime' in str(s.dtype):
        return 'Time'
    elif s.dtype in [np.bool] or len(set(s.dropna().unique()) - set([False,True]))==0:
        return 'Boolean'
    else:
        return 'Categorical'

def get_feature_info(df,feature):
    if feature in df.columns:
        cardinality = df[feature].nunique()
        missed= 100 * df[feature].isnull().sum() / df.shape[0]
        feature_type = get_feature_type(df[feature])
        return [feature_type,cardinality,missed]
    else:
        return '','',''

def get_categorical(df,maxmissed=0.6,binary=True):
    """Select features for cramer v plot  - categorical or binary,
       features with too many different values are ignored
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','datetime64','m8[ns]']

    #keep columns with % of missed less then 60
    categoricals = df.loc[:, df.isnull().mean() <= maxmissed].select_dtypes(exclude=numerics).columns.to_list()

    if binary:
        #add binary columns
        bool_cols = [col for col in df.select_dtypes(include=numerics).columns.to_list() if
                   df[col].dropna().value_counts().index.isin([0,1]).all()]

        categoricals.extend(bool_cols)

    #drop columns with no variance and with too much variance (id etc)
    categoricals=[col for col in categoricals if
               df[col].dropna().nunique() >1 and df[col].nunique() < df.shape[0]/2]

    return categoricals
