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

warnings.filterwarnings("ignore")

#=====================#=====================#=====================#=====================
# explore datasets categorical variables
#=====================#=====================#=====================#=====================

def anova_explore_dataset(df,target,max_card=200):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        df  (DataFrame): pandas DataFrame
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """    
    features = list(set(df.columns.to_list()))
    if target not in features:
        print("{} not in dataframe".format(target))
        return
    
    
    for feature in features:
                                                                   
        if feature==target:
            continue                                                                  
                                                                   
        feature_type,cardinality,missed = get_feature_info(df,feature)
        
        if (feature_type=='Categorical' and cardinality<max_card and cardinality>1 and cardinality< df.shape[0]/2.0 ) or feature_type=='Boolean':
            
            display(HTML("<h3 align=\"center\">{}</h3>".format(feature)))
            
            if(anova(df[[feature,target]].dropna(),feature,target)):
                
                # Tukey HSD test
                # target mean on feature values
                comp = mc.MultiComparison(df[target],df[feature].fillna('MISSED'))
                post_hoc_res = comp.tukeyhsd()
                results_as_html =post_hoc_res.summary().as_html()
                results_as_pandas = pd.read_html(results_as_html, header=0, index_col=0)[0]
                print( results_as_pandas[results_as_pandas.reject].sort_values('meandiff',ascending=False) if results_as_pandas.shape[0]>0 and results_as_pandas[results_as_pandas.reject].shape[0]>0 else results_as_pandas  )              

#=====================#=====================#=====================#=====================
# compare datasets 
#=====================#=====================#=====================#=====================

def anova_compare_datasets(df1,df2,target,max_card=200):
    df1['flag176542']=0
    df2['flag176542']=2
    df=pd.concat([df1,df2])
    _compare_datasets(df,'flag176542',target,max_card)
    
def _compare_datasets(df,flag,target,max_card=200):
    
    model = ols(target+' ~ C('+flag+')', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print('Anova ' ,anova_table['PR(>F)'].iloc[0], 'OK' if anova_table['PR(>F)'].iloc[0] <0.05 else 'KO')
    
    #Shapiro-Wilk test to check the normal distribution of residuals
    w, pvalue = stats.shapiro(model.resid)
    print('\nShapiro-Wilk test  normal distribution of residuals .  ', 'OK' if (pvalue <0.05) else 'KO' )
    print(w, pvalue)
    
    # Bartlett’s test to check the Homogeneity of variances
    w, pvalue = stats.bartlett(df[df[flag]==0][target],df[df[flag]==1][target])
    print('\nBartlett’s test', 'OK' if pvalue <0.05 else 'KO')
    print(w, pvalue)

    # stats f_oneway functions takes the groups as input and returns F and P-value
    #fvalue, pvalue = stats.f_oneway(df[df[flag]==0][target],df[df[flag]==1][target] )
    
    #print('Anova ' ,fvalue, pvalue, 'OK' if pvalue <0.05 else 'KO')
    if (anova_table['PR(>F)'].iloc[0]>0.05):
        return
    
    sns.boxplot(x=flag, y=target, hue=flag, data=df, palette="Set3") 
    pls.show()
    
    features = list(set(df.columns.to_list()))

    for feature in features:
                                                                   
        if feature==target:
            continue                                                                  
                                                                   
        feature_type,cardinality,missed = get_feature_info(df,feature)
        
        if (feature_type=='Categorical' and cardinality<max_card) or feature_type=='Boolean':
            display(HTML("<h3 align=\"center\">{}</h3>".format(feature)))
            res=_compare_datasets_by_feature_anova(df,flag,feature,target,False)     
            if res.shape[0]>1:
                print('Anova test:')
                print(res,'\n\n')

def _compare_datasets_by_feature(df,flag,feature,target):
    
    resus=pd.DataFrame()
    for f in df[feature].unique():
        
        try:
            res = turkeyHSD(df[df[feature]==f],flag,target)
        except:
            print('Failed',f)
        else:
            if res.shape[0]>0:
                res[feature]=f
                resus=pd.concat([resus,res])
    return  resus[resus.reject].sort_values('meandiff',ascending=False) if resus.shape[0]>0 and resus[resus.reject].shape[0]>0 else resus

def _compare_datasets_by_feature_anova(df,flag,feature,target,verbose=True):
    resus=pd.DataFrame()
    d=[]
    v1=[]
    v2=[]
    p=[]
    for f in df[feature].unique():
        a=df[(df[flag]==0)&(df[feature]==f)][target]
        b=df[(df[flag]==1)&(df[feature]==f)][target]
        #anova
        #fvalue, pvalue = stats.f_oneway(a,b )
        try:
            model = ols(target+' ~ C('+flag+')', data=df[df[feature]==f]).fit()
            anova_table = sm.stats.anova_lm(model, typ=2) 
            if verbose:
                print(f,' anova PR(>F) ',anova_table['PR(>F)'].iloc[0])
        except:
            a=6
            if verbose:
                print (f," Failed")
        else:
            if anova_table['PR(>F)'].iloc[0]>0.05:
                continue

            w, pvalue = stats.bartlett(a,b)
            
            if pvalue>0.05:
                continue

            d.append(f)
            p.append(anova_table['PR(>F)'].iloc[0])
            v1.append(a.mean())
            v2.append(b.mean())  

    res=pd.DataFrame({'feature':d,'1':v1,'2':v2,'pvalue':p})

    res['diff']=abs(res['1']-res['2'])

    return res.sort_values('diff',ascending=False)[['feature','diff','pvalue']] if res.shape[0]>0 else res


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
    return results_as_pandas[results_as_pandas.reject].sort_values('meandiff',ascending=False) if results_as_pandas.shape[0]>0 and results_as_pandas[results_as_pandas.reject].shape[0]>0 else results_as_pandas


def anova(df,feature,target,verbose=True):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        df  (DataFrame): pandas DataFrame
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """    
    model = ols(target+' ~ C('+feature+')', data=df).fit()
    
    #Shapiro-Wilk test to check the normal distribution of residuals
    w, pvalue = stats.shapiro(model.resid)
    if (verbose):
        print('\nShapiro-Wilk test - normal distribution of residuals . ', 'OK' if (pvalue <0.05) else 'KO' )
        print(w, pvalue)
    if (pvalue>0.05):
        return False
    
    anova_table = sm.stats.anova_lm(model, typ=2)
    if (verbose):
        print(anova_table)
        print('Anova ' ,anova_table['PR(>F)'].iloc[0], 'OK' if anova_table['PR(>F)'].iloc[0] <0.05 else 'KO')
        
    return (anova_table['PR(>F)'].iloc[0]<0.05)

#=====================#=====================#=====================#=====================
# two way anova
#=====================#=====================#=====================#=====================

def two_way_anova(df,feature1,feature2,target):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        df  (DataFrame): pandas DataFrame
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """    
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
        return "","",""            

    