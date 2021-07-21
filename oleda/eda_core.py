import numpy as np
import matplotlib.pyplot as pls 
import pandas as pd

import shap
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier,LGBMRegressor

from IPython.display import display, HTML

#=====================#=====================#=====================#=====================
# shap 
#=====================#=====================#=====================#=====================

def plot_shap(x, target,ignore=[],nbrmax=20):
    
    features=x.columns.to_list()
    features.remove(target)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    #doesn't work on time columns, remove id columns (all values are different), columns with all nulls 
    for f in x.columns.to_list():
        
        if (isTime(x[f].dtype) or x[f].isnull().values.all() or (len(x[f].unique())>x.shape[0]/2.0 and str(x[f].dtype) not in numerics))  and f in features:
            features.remove(f)
            
    features=list(set(features)-set(ignore))

    #list of categorical features
    categorical_features=x[features].select_dtypes(exclude=numerics).columns.to_list()

    #change type to categorical for lightgbm
    backup={}
    for c in categorical_features:
        backup[c]=x[c].dtype
        x[c] = x[c].astype('category')

    target_type,target_cardinality,_=get_feature_info(x,target)
    binary_target=(target_type=='Numeric' and target_cardinality==2)
    
    if binary_target:
        clf = LGBMClassifier(
                             objective='binary'
                             ,n_estimators=100
                            , min_data_in_leaf = 10
                            , min_sum_hessian_in_leaf = 10
                            , feature_fraction = 0.9
                            , bagging_fraction = 1
                            , bagging_freq = 1                     
                            , metric='auc'
                            , learning_rate = 0.03
                            , num_leaves = 19
                            , num_threads = 2
                            , nrounds = 500 
                            )
    else:
        clf = LGBMRegressor(                           
                             n_estimators=100
                            , min_data_in_leaf = 10
                            , min_sum_hessian_in_leaf = 10
                            , feature_fraction = 0.9
                            , bagging_fraction = 1
                            , bagging_freq = 1                     
                            , learning_rate = 0.03
                            , num_leaves = 19
                            , num_threads = 2
                            , nrounds = 500 
                            )
    clf.fit(x[features], x[target])#,categorical_feature=categorical_features)
    
    shap_values = shap.TreeExplainer(clf.booster_).shap_values(x[features])
    shap.summary_plot(shap_values, x[features], max_display=nbrmax, auto_size_plot=True)
    
    if binary_target:
        vals= np.abs(shap_values).mean(0)
    else:
        vals= shap_values
        
    feature_importance = pd.DataFrame(list(zip(x[features].columns, sum(vals))), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    sorted_features=feature_importance['col_name'].to_list()

    X=x.copy()

    for f in categorical_features:
        X[f]=  X[f].astype(object)
        X[f]=pd.factorize(X[f])[0]     
       
    if binary_target:
        shap.summary_plot(shap_values[1], x[features])        
        
        for name in sorted_features[:nbrmax]:
            if name in categorical_features and x[name].astype(str).nunique()>100:
                continue
            fig, ax = pls.subplots(1,1,figsize=(20,10))
            shap.dependence_plot(name, shap_values[1], X[features], display_features=x[features], interaction_index=None,ax=ax)
            pls.show()

    #restore type
    for c in categorical_features:
        x[c] = x[c].astype(backup[c])
        
    return sorted_features

#=====================#=====================#=====================
# time series plots
#=====================#=====================#=====================

def plot_ntop_categorical_values_(df,feature,target, nbr_max, figsize, linewidth, period, method_name,sample=False):
    
    if sample:
        values=df[feature].value_counts().sample(nbr_max).index.to_list()
    else:
        values=df[feature].value_counts()[:nbr_max].index.to_list()
        
    if  len(values)==0:
        return
    
    resampler=df[df[feature]==values[0]][target].resample(period)
    ax=getattr(resampler,method_name)().plot(x_compat=True,figsize=figsize, grid=True,linewidth=linewidth)
    legend=[values[0]]
    
    for i in range(1,len(values)):
        resampler=df[df[feature]==values[i]][target].resample(period)
        getattr(resampler,method_name)().plot(x_compat=True, figsize=figsize,ax=ax, grid=True, linewidth=2.0,title='{} per day'.format(feature))
        legend.append(values[i])

    if len(values) > 1:
        ax.lines[1].set_linestyle(":")
    ax.lines[0].set_linestyle("--")
    pls.legend(legend, bbox_to_anchor=(1.2, 0))
    pls.show()
    
# plots n top feature values counts per day   
def plot_ntop_categorical_values_counts(df,feature,target, nbr_max=4, figsize=(20,4), linewidth=2.0, period="1d",sample=False):

    plot_ntop_categorical_values_(df,feature,target, nbr_max, figsize, linewidth, period, 'count',sample)
    return
    
    
def plot_ntop_categorical_values_sums(df,feature,target,nbr_max=4,figsize=(20,4),linewidth=2.0,period="1d",sample=False):
    
    plot_ntop_categorical_values_(df,feature,target, nbr_max, figsize, linewidth, period, 'sum',sample)
    return

    
def plot_ntop_categorical_values_means(df,feature,target,nbr_max=4,figsize=(20,4),linewidth=2.0,period="1d",sample=False):
    
    plot_ntop_categorical_values_(df,feature,target, nbr_max, figsize, linewidth, period, 'mean',sample)
    return    

    
#=====================#=====================#=====================#=====================
# cramers V
#=====================#=====================#=====================#=====================
#Theil’s U, conditional_entropy (no symetrical)
#https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
#https://github.com/shakedzy/dython/blob/master/dython/nominal.py
   
import scipy.stats as ss
import itertools

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
def cramer_v_corr(df,categoricals,ax=None,figsize=(10,10)):
    print(categoricals)
    fig=None
    if ax==None:
        fig, ax = pls.subplots(1,1,figsize=figsize)
        
    correlation_matrix = pd.DataFrame(
        np.zeros((len(categoricals), len(categoricals))),
        index=categoricals,
        columns=categoricals
    )

    for col1, col2 in itertools.combinations(categoricals, 2):
        print(col1, col2 )
        idx1, idx2 = categoricals.index(col1), categoricals.index(col2)
        correlation_matrix.iloc[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        correlation_matrix.iloc[idx2, idx1] = correlation_matrix.iloc[idx1, idx2]

    ax = sns.heatmap(correlation_matrix, annot=True, ax=ax); 
    ax.set_title("Cramer V Correlation between Variables");

    if fig!=None:#local figure
        pls.show()

def get_categorical(df,ignore=[]):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','datetime64','m8[ns]'] 
    
    #keep columns with % of missed less then 60
    categoricals = df.loc[:, df.isnull().mean() <= .6].select_dtypes(exclude=numerics).columns.to_list()
    
    #add binary columns
    bool_cols = [col for col in df.select_dtypes(include=numerics).columns.to_list() if 
               df[col].dropna().value_counts().index.isin([0,1]).all()]
    
    categoricals.extend(bool_cols)
    
    #drop columns with no variance and with too much variance (id etc) 
    categoricals=[col for col in categoricals if 
               df[col].dropna().nunique() >1 and df[col].nunique() < df.shape[0]/2]
    
    
    return list(set(categoricals)-set(ignore))


def plot_cramer_v_corr(df,max_features=20,ax=None,ignore=[]):
    # plot features correlation (Theil’s U, conditional_entropy) heatmap
    #max_features max features to display
    #features are selected automaticly - categorical or binary  
    #features with too many different values are ignored
    
    categorical=get_categorical(df,ignore)[:max_features]

    if len(categorical)>1:
        cramer_v_corr(df,categorical,ax)

    #Theil’s U, conditional_entropy (no symetrical)
    #https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    #https://github.com/shakedzy/dython/blob/master/dython/nominal.py
    #https://stackoverflow.com/a/46498792/5863503
    
#=====================#=====================#=====================#=====================
# nan
#=====================#=====================#=====================#=====================

def missing_values_table(df):
    
    missing = df.isnull().sum()
    percent = 100 * missing / len(df)
    missing = pd.concat([missing, percent], axis=1)
    missing.columns = ['Missing', '% of Total']
    missing = missing[missing.iloc[:,1] != 0].sort_values('% of Total', ascending=False).round(1)

    return missing   

#=====================#=====================#=====================#=====================
# html
#=====================#=====================#=====================#=====================

def header(title,sz='h2'):
    print('\n \n')
    display(HTML("<hr>"))
    display(HTML("<{} align=\"center\">{}</{}>".format(sz,title,sz)))
    print('\n  ') 
    
#=====================#=====================#=====================#=====================
# features 
#=====================#=====================#=====================#=====================

def get_feature_type_(dtype):
    if dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        return 'Numeric'
    elif isTime(dtype):
        return 'Time'
    elif dtype in [np.bool]:
        return 'Boolean'
    else:
        return 'Categorical'   
        
def get_feature_type(s):
    if s.dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        return 'Numeric'
    elif isTime(s.dtype):
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
    
#=====================#=====================#=====================#=====================
# to find time columns 
#=====================#=====================#=====================#=====================
def isTime(dtype):
    if '[ns]' in str(dtype) or 'datetime' in str(dtype) :   
        return True
    return False

def safe_convert(s):
    try:
        return pd.to_datetime(s, errors='ignore') 
    except:
        a=0
    return s