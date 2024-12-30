"""Exploratory Data Analysis with python.

   Automated report generation from a pandas DataFrame for uncovering
   insights within the data.

   Typical usage example:
         import oleda
         oleda.report(df)
"""
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings('ignore')

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as pls

import shap
import seaborn as sns
from scipy import stats
from IPython.display import display, HTML
from lightgbm import LGBMClassifier,LGBMRegressor

from .eda_anova import anova,two_way_anova,turkeyHSD


#=====================#=====================#=====================
# single dataset  eda
#=====================#=====================#=====================

def report(df,target=None,**kwarg):
    """Explore single dataset and create report.

    The report includes:
        Statistics on missing values
        Information on each feature relevant to its type
        Pearson correlation heatmap
        Cramers V statistics
        Pair plot for the most correlated features
        Time series plots are included if the dataframe index is a valid 
        datetime index.
        The dataset can be tested against a target variable (binary (0,1)
        or continuous):
            oleda.report(df,target,ignore=[],nbrmax=20)
        in this case, plots showing the correlation with the target are added 
        for each feature. Features are sorted according to their importance 
        by SHAP. The number of most important features selected by SHAP 
        are explored.
        Features to be ignored can be added to the ignore list.

    Args:
        df       (object): Pandas dataframe
    Keyword Args:
        dependency (bool): Set to True to plot dependency plots
        maxcount   (int) : Maximum number of features to display
        maxshap    (int) : Maximum number of shapley values to display
        ignore     (list): List of features to ignore
    """

    ignore=kwarg.get('ignore',[])
    kwarg['maxshap']=kwarg.get('maxshap',df.shape[1])

    if (target is not None) and (target not in df.columns.to_list()):
        print(f'{target} is not in dataframe - Ignore')
        target=None

    #detect time columns
    df = df.apply(lambda col: safe_convert(col)
                  if col.dtypes == object else col, axis=0)

    #print nans statistics
    header('Missed values' )
    print_na(df,**kwarg)

    if (target is not None) and is_numeric(df[target].dtype) :
        header('Shap values')
        sorted_features=plot_shaps(df,target,**kwarg)
    else:
        sorted_features=[f for f in df.columns.tolist() if f not in ignore]

    # if dataframe has timedate index - plot targets time series
    if target is not None and is_time(df.index.dtype) and df.index.nunique()> 2:
        header('Time series' )
        ax=df[target].resample('1d').mean().plot(
            grid=True,x_compat=True,figsize=(20,4),linewidth=2.0,label=target)
        pls.title(' {target} mean per day ')
        pls.legend()
        pls.show()

    header('Features' )
    print_features_stats(df,target,sorted_features)

    #for numeric variables only
    header('Pearson correlations' )
    plot_corr(df,**kwarg)

    #correlations of categorical variables
    header('Cramers V staticstics' )
    plot_cramer_v_corr(df,**kwarg)

    #plot 10 most correlated features
    header('Top correlated features' )
    f=get_top_correlated(df,th=0.099,maxcount=7)
    if (target is not None) and (target not in f):
        f.append(target)
    g=sns.pairplot(df[f])

    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation = 90)
        ax.set_ylabel(ax.get_ylabel(), rotation = 0)
        ax.yaxis.get_label().set_horizontalalignment('right')

#=====================#=====================#=====================#
# shap
#=====================#=====================#=====================#
def plot_shaps(df, target, **kwarg):
    """Calculate Shapley values.

    This function builds a LightGBM model to predict the target 
    variable and then calculates and plots its Shapley values.

    Args:
        df   (DataFrame): pandas DataFrame.
        target     (str): Name of the dependent feature.

    Keyword Args:
        dependency (bool): Set to True to plot dependency plots.
        maxshap    (int) : Maximum number of features to display.
        maxcount   (int) : Maximum number of features to display 
                          (used if maxshap is not set)
        ignore     (list): List of features to ignore.

    Returns:
        Feature names sorted by importance.
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    maxcount  = kwarg.get('maxshap', kwarg.get('maxcount',df.shape[1]))
    ignore    = kwarg.get('ignore',[])

    #doesn't work on time columns, remove id columns (all values are different),
    #columns with all nulls
    filterout=lambda f: (is_time(df[f].dtype)
                         or df[f].isnull().values.all()
                         or  (len(df[f].unique())>df.shape[0]/2.0
                              and str(df[f].dtype) not in numerics))

    features = [ f for f in df.columns if ~filterout(f) ]
    features.remove(target)

    features=list(set(features)-set(ignore))
    _=[print(f'Feature name {x} cantains special JSON characters - Skip')
       for x in features if ':' in x ]
    features=[ x for x in features  if not ':' in x  ]

    #list of categorical features
    categorical_features=df[features].select_dtypes(exclude=numerics).columns.to_list()

    #change type to categorical for lightgbm
    backup={}
    for c in categorical_features:
        backup[c]=df[c].dtype
        df[c] = df[c].astype('category')

    binary_target=((str(df[target].dtype) in numerics) and (df[target].nunique()==2))

    parameters={              'n_estimators':100
                            , 'min_data_in_leaf' : 10
                            #, 'min_sum_hessian_in_leaf' : 10
                            , 'feature_fraction' : 0.9
                            , 'bagging_fraction' : 1
                            , 'bagging_freq' : 1
                            , 'learning_rate' : 0.03
                            , 'num_leaves' : 19
                            , 'num_threads' : 2
                            , 'nrounds' : 500 }

    if binary_target:
        parameters['objective']='binary'
        parameters['metric']='auc'
        clf = LGBMClassifier(**parameters)
    else:
        clf = LGBMRegressor( **parameters )
    clf.fit(df[features], df[target])#,categorical_feature=categorical_features)

    shap_values = shap.TreeExplainer(clf.booster_).shap_values(df[features])
    if type(shap_values) is list:
        shap_values=shap_values[1]
        
    shap.summary_plot(shap_values, df[features], max_display=maxcount)#, auto_size_plot=True

    vals= shap_values

    feature_importance = pd.DataFrame(list(zip(df[features].columns, sum(abs(vals)))),
                                      columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    sorted_features=feature_importance['col_name'].to_list()

    if binary_target:
        
        #shap.summary_plot(shap_values[:,:maxcount], df[features[:maxcount]])

        if kwarg.get('dependency',False):
            
            data=df.copy()#do not copy !!!
            
            for f in categorical_features:
                data[f]=  data[f].astype(object)
                data[f]=pd.factorize(data[f])[0]

            for name in sorted_features[:maxcount]:
                #continue
                if name in categorical_features and df[name].astype(str).nunique()>100:
                    continue
                fig, ax = pls.subplots(1,1,figsize=(20,10))
                shap.dependence_plot(name, shap_values[1], data[features],
                                     display_features=df[features], interaction_index=None,ax=ax)
                pls.show()

    #restore type
    for c in categorical_features:
        df[c] = df[c].astype(backup[c])

    return sorted_features

#=====================#=====================#=====================#=====================
# numerical continues
#=====================#=====================#=====================#=====================

def plot_cuts(df,feature,target,**kwarg):

    """Split continuous variable into bins and plot statistics per bin.

    This function splits a continuous variable into bins and plots the target 
    mean and count for each bin. If the bins argument is not provided, 
    the variable is split into nbins bins.

    Args:
        df  (DataFrame): Pandas DataFrame.
        feature   (str): Feature to split.
        target    (str): Feature to calculate statistics for.

    Keyword Args:
        figsize (tuple): Size of the plot.
        nbins     (int): Number of bins.
        bins     (list): List of specific bin edges.
    """

    figsize=kwarg.get('figsize',(8,4))
    bins=kwarg.get('bins',None)
    if bins is None:
        bins=np.arange(df[feature].min(),df[feature].max(),
                       (df[feature].max()-df[feature].min())/(1+kwarg.get('nbins',10)))

    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title(f'Histogram of {feature}')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('count')
    ax2.set_xlabel(feature)
    ax2.set_ylabel(target)
    df.groupby(pd.cut(df[feature], bins=bins))[target].count().plot(kind='bar',ax=ax1,grid=True)
    df.groupby(pd.cut(df[feature], bins=bins))[target].mean( ).plot(kind='bar',ax=ax2,grid=True)
    pls.show()

def plot_qcuts(df,feature,target,**kwarg):

    """Split a continuous variable into quantiles and plot statistics per quantile.

    This function splits a continuous variable into quantiles and plots the target mean
    for each quantile. If the qbins argument is not provided, the variable is split into
    10 quantiles by default.

    Args:
        df  (DataFrame): Pandas DataFrame.
        feature   (str): Feature to split.
        target    (str): Feature to calculate statistics for.

    Keyword Args:
        figsize (tuple) : Size of the plot.
        qbins   (list)  : List of specific quantile values.
    """

    figsize=kwarg.get('figsize',(4,4))
    qbins=kwarg.get('qbins', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    fig = pls.figure( figsize=figsize)
    pls.title(f'Histogram of {feature}')
    pls.xlabel(feature)
    pls.ylabel(target)

    sub=df.groupby(pd.qcut(df[feature], q=qbins,duplicates='drop'))[target]
    getattr(sub,kwarg.get('method_name','mean'))().plot(kind='bar',grid=True)

    pls.show()

def plot_qcuts3x(df,feature1,feature2,target,**kwarg):

    """Split continuous variables into quantiles and plot statistics per quantile.

    This function splits continuous variables into quantiles and plots distribution 
    of target variable according to obtained levels.
    If the qbins argument is not provided, the variable is split into
    10 quantiles by default.

    Args:
        df  (DataFrame): Pandas DataFrame.
        feature   (str): Feature to split.
        target    (str): Feature to calculate statistics for.

    Keyword Args:
        figsize (tuple) : Size of the plot.
        qbins   (list)  : List of specific quantile values.
    """

    figsize=kwarg.get('figsize',(14,8))
    qbins=kwarg.get('qbins', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    fig = pls.figure( figsize=figsize)
    tmp=df[[feature1,feature2,target]].copy()

    tmp[feature1]=pd.qcut(tmp[feature1], q=qbins, duplicates='drop')
    tmp[feature2]=pd.qcut(tmp[feature2], q=qbins, duplicates='drop')
    tmp=tmp.sort_values([feature1,feature2])
    ax=sns.boxplot(x=feature1, y=target, hue=feature2, data=tmp)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    pls.show()
#=====================#=====================#=====================#=====================
# categorical
#=====================#=====================#=====================#=====================
def plot_stats(df,feature,target,**kwarg):

    """Plot target statistcs per feature value.

    Plot target mean and count per categorical feature value.

    Args:
        df  (DataFrame): Pandas DataFrame.
        feature   (str): Feature to group data by.
        target    (str): Feature to calculate statistics for.

    Keyword Args:
        ax         (list): List of matplotlib axes subplots.
        maxcount   (int) : Maximum number of feature values to display. Default value is 20.
        figsize    (list): Size of the figure.
        sortby     (str) : Column name to sort by.
    """

    maxcount=kwarg.get('maxcount',20)
    ax=kwarg.get('ax',None)
    sort=kwarg.get('sortby','Count ')

    cat_count = df[feature].value_counts().reset_index()
    cat_count.columns = [feature,'Count ']

    cat_perc = df[[feature, target]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_count,on=feature)
    cat_perc.sort_values(by=sort, ascending=False, inplace=True)

    size=kwarg.get('figsize',(12,6) if len(cat_count[:maxcount]) <=40 else (12,14))
    if ax is None:
        fig, ax = pls.subplots(ncols=2, figsize=size)

    sns.set_color_codes('pastel')
    s = sns.barplot(ax=ax[0], x = feature, y='Count ',
                    order=cat_perc[feature][:maxcount],data=cat_perc[:maxcount])
    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    s = sns.barplot(ax=ax[1], x = feature, y=target,
                    order=cat_perc[feature][:maxcount], data=cat_perc[:maxcount])
    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    pls.ylabel(target, fontsize=10)
    pls.tick_params(axis='both', which='major', labelsize=10)

    if kwarg.get('showfigure',True):
        pls.show()

def plot_melt(df,feature,target1,target2,**kwarg):
    """Plots two targets statistcs per feature value.

    Plots two targets mean and count for categorical feature value.

    Args:
        df  (DataFrame): Pandas DataFrame.
        feature   (str): Feature to group data by.
        target1   (str): Feature to calculate statiscs for.
        target2   (str): Feature to calculate statiscs for.

    Keyword Args:
        maxcount   (int) : Maximum number of feature values to display.
    """
    end=kwarg.get('maxcount',20)
    cat_count = df[feature].value_counts().reset_index()
    cat_count.columns =[feature,'Count ']
    cat_count.sort_values(by='Count ', ascending=False, inplace=True)

    cat_perc = df[[feature, target1,target2]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_count,on=feature)

    cat_perc.sort_values(by='Count ', ascending=False, inplace=True)
    cat_perc=cat_perc[:end]

    data_melted = pd.melt(cat_perc[[feature,target1,target2]], id_vars=feature,
                           var_name='source', value_name='value_numbers')

    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=(12,6))
    sns.set_color_codes('pastel')
    s = sns.barplot(ax=ax1, x = feature, y='Count ',
                    order=cat_count[feature][:end],data=cat_count[:end])
    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    s = sns.barplot(ax=ax2,
                    x = feature,
                    y='value_numbers',
                    hue='source',
                    order=data_melted[feature][:min(end,cat_count.shape[0])],
                    data=data_melted)

    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    pls.tick_params(axis='both', which='major', labelsize=10)
    pls.legend(bbox_to_anchor=(2, 0))
    pls.show()

#==========================================#=====================#=====================
# nan
#==========================================#=====================#=====================

def plot_na(df,** kwarg):
    """Plot the percentage of missing values for each column.

    Args:
        df  (DataFrame): Pandas DataFrame.

    Keyword Args:
        figsize (tuple): Size of the figure.
    """
    #pls.style.use('seaborn-talk')
    figsize=kwarg.get('figsize',(18,6))
    fig = pls.figure(figsize=figsize)
    miss = pd.DataFrame((df.isnull().sum())*100/df.shape[0]).reset_index()
    ax = sns.pointplot(x='index',y=0,data=miss)
    pls.xticks(rotation =90,fontsize =7)
    pls.title('Missed data')
    pls.ylabel(' %')
    pls.xlabel('Features')

def print_na(df, **kwarg):
    """Print the percentage of missing values for each column.

    Args:
        df  (DataFrame): Pandas DataFrame.

    Keyword Args:
        maxcount (int) : Maximum number of features to display.
    """
    mdf=missing_values_table(df)
    if mdf.shape[0]:
        print(missing_values_table(df).head(kwarg.get('maxcount',df.shape[0])))
    else:
        print('No missed values in dataframe ')

def get_top_correlated(df,th=0.01,maxcount=10):
    """Returns maxcount most correlated numerical features.

    Find all numerical features and select most correlated features names.

    Args:
        df  (DataFrame): pandas DataFrame
        maxcount  (int): Maximum number of features to return.

    Returns:
        A list of the names of the most correlated features.
    """
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    corr=df[numeric].corr()
    if corr.shape[0]>0:
        cx=corr.unstack().dropna().abs().sort_values(ascending=False).reset_index()
        return list(cx[(cx[0]>th)&(cx.level_0!=cx.level_1)
                      ]['level_0'].drop_duplicates(keep='first')[:maxcount].values)
    else:
        return []

def plot_corr(df, **kwargs):
    """Plot correlation heatmap for all numerical features.

    Identify and plot the correlation heatmap for all numerical 
    features in the provided Pandas DataFrame.

    Args:
        df  (DataFrame): Pandas DataFrame.

    Keyword Args:
        maxcount     (int): Maximum number of features to display.
        figsize    (tuple): Size of the heatmap.
        features    (list): List of features to investigate.
        features_ext(list): Second list of features to calculate correlation w
                            ith the first one.

    """
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    features=kwargs.get('features',numeric)
    features_ext=kwargs.get('features_ext',features)

    maxcount=kwargs.get('maxcount',len(numeric))

    corr=df[numeric].corr()
    cx=corr.abs().unstack().sort_values(ascending=False).reset_index().dropna()
    if 'features' not in kwargs:
        features=list(cx[cx.level_0!=cx.level_1]['level_0']
                      .drop_duplicates(keep='first')[:maxcount].values)
        features_ext=features

    if (len(features)+len(features_ext))<2:
        return

    v=min(len(features),maxcount,20)
    figsize=kwargs.get('figsize',(v,v))

    fig, ax = pls.subplots(1,1,figsize=figsize)
    sns.heatmap( corr[corr.index.isin(features)][features_ext],
                cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    pls.title('Correlation Heatmap')
    pls.show()
    return


#=====================#=====================#=====================#=====================
# report
#=====================#=====================#=====================#=====================
def print_features_stats(df,target=None,sorted_features=None):
    """Plot per feature statistics.

       Plot statistics for each feature from sorted_features list if it is not NULL or
       for each feature from dataframe otherwise.

    Args:
        df  (DataFrame): Pandas DataFrame
        target    (str): Feature to predict.
        sorted_features   (str): Feature to print statistics for. Optional

    """
    #explore features selected by shap (sorted_features)
    features = sorted_features if len(sorted_features)>0 else df.columns.to_list()
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    numeric=list(set(numeric)&set(sorted_features))

    for feature in features:

        if feature==target:
            continue

        header(feature,'h3','\n ')

        feature_type,cardinality,missed = get_feature_info(df,feature)

        #all nans
        if cardinality<1:
            print ('No values but nan')
            continue

        info = pd.DataFrame(index=['Type :' ,'Distinct count :', 'Missed %:'], columns=[' '])
        info[' ']=[feature_type,cardinality,missed ]
        print(info.head())
        print('\n ')

        if feature_type=='Categorical' or feature_type=='Boolean':
            explore_categorical(df,feature,target,numeric)
        elif feature_type=='Numeric':
            explore_numerical (df,feature,target)
        else:
            explore_timecolumn(df,feature)

def explore_timecolumn(df,feature):
    info = pd.DataFrame(index=['dType :' ,'Min :', 'Max :'],columns=[' '])
    info[' ']=[df[feature].dtype,df[feature].min(),df[feature].max()]
    print(info.head())
    print('\n ')
    print('Time column skip plotting ')

def explore_categorical(df,feature,target,numeric,maxcount=30,maxtscount=4):

    df[feature]=df[feature].fillna('Missed value')

    tg_cardinality=df[target].nunique() if target is not None else 0
    cardinality=df[feature].nunique()

    if cardinality > df.shape[0]/2.0 :
        print('Too many values to plot ')
    elif df[feature].isnull().values.all() :
        print('All values are null')
    elif cardinality<2:
        print('Zero variance')
    else:
        if target is not None :
            plot_stats(df,feature,target,maxcount=maxcount)
        else:
            fig,ax =  pls.subplots(1, 1,figsize=(9, 5))
            if cardinality>maxcount:
                f=df[feature].value_counts()[:maxcount].index
                df[df[feature].isin(f)][feature].hist()
            else:
                df[feature].astype('str').hist()
            pls.xticks(rotation='vertical')
            pls.show()

        #count of records with feature=value per day
        if target is not None and  is_time(df.index.dtype) and df.index.nunique()>2:
            display(HTML(f'<h3 align=\"center\">Top {feature} count per day</h3>'))
            plot_ntop_categorical_values(df,feature,target,maxcount=maxtscount,method_name='count')

            #mean of target for records with feature=value per day
            display(HTML(f'<h3 align=\"center\">{target} mean per day </h3>'))
            plot_ntop_categorical_values(df,feature,target,maxcount=maxtscount,method_name='mean')

        if target is not None and tg_cardinality>2 and cardinality<maxcount :
            sns.catplot(y=feature,x=target,data=df, orient='h', kind='box')
            pls.show()

    #partitioning
    if (len(numeric)>1) and (cardinality<maxcount) and (cardinality>1):
        depended =[]

        #to speed up - select 4 most popopular values
        top=df[feature].value_counts().iloc[:maxtscount].index
        sub=df[df[feature].isin(top)]

        for t in numeric:
            try:
                #check variance
                if anova(sub,feature,t,False):
                    depended.append(t)
            except Exception:
                pass

        if len(depended)>0:
            print('Anova passed for ',depended)
            if len(depended)>1:
                depended=get_top_correlated(sub[depended])
                if len(depended)>1:
                    sns.pairplot(sub,vars=depended,hue=feature,corner=True)
                    pls.show()

    df[feature]=df[feature].replace('Missed value',np.nan)

def explore_numerical(df,feature,target,maxcount=40):

    tg_cardinality= 0 if target is None else df[target].nunique()
    cardinality=df[feature].nunique()

    info = pd.DataFrame(index=['dType :' ,'Min :', 'Max :', 'Mean :', 'Std :'],
                                                                 columns=[' '])
    info[' ']=[df[feature].dtype,
               df[feature].min(),
               df[feature].max(),
               df[feature].mean(),
               df[feature].std() ]

    print(info.head())
    print('\n ')

    #pairwise_feature_sum_per_day(df1,df2,feature)
    #pairwise_feature_mean_per_day(df1,df2,feature)
    if cardinality<=maxcount:
        if target is not None :
            plot_stats(df,feature,target,maxcount=maxcount)
        else:
            pls.hist(df[feature])
            pls.show()

        if target  is not None and tg_cardinality>2 and cardinality>2 :
            fig,ax =  pls.subplots(1, 1,figsize=(8, 5))
            pls.scatter(df[feature], df[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            pls.show()

        if target  is not None and tg_cardinality>2:
            sns.catplot(y=feature,x=target,data=df, orient='h', kind='box')
            pls.show()

    else:
        #distribution
        fig,ax = pls.subplots(1, 2,figsize=(16, 5))
        sns.distplot(df[feature],kde=True,ax=ax[0])
        ax[0].axvline(df[feature].mean(),color = 'k',linestyle='dashed',label='MEAN')
        ax[0].legend(loc='upper right')
        ax[0].set_title(f'Skewness = {df[feature].skew():.4f}')
        sns.boxplot(x=df[feature],color='blue',orient='h',ax=ax[1])
        pls.show()

        if  target  is not None :
            if tg_cardinality < 10:
                fig,ax = pls.subplots(1, 2,figsize=(16, 5))
                #ax[0].scatter(df[feature], df[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
                #ax[0].set_xlabel(feature)
                #ax[0].set_ylabel(target)
                sns.kdeplot(data=df, x=feature, hue=target, multiple='stack' ,ax=ax[0])
                sns.violinplot(x=target, y=feature, data=df, ax=ax[1], inner='quartile')
                pls.show()
            else:
                #continues - continues
                plot_qcuts(df,feature,target)
                q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                df['cuts__'+feature]=pd.qcut(df[feature],q=q,duplicates='drop')
                sns.catplot(x=target,y='cuts__'+feature,data=df, orient='h', kind='box')
                pls.show()
                del df['cuts__'+feature]
                fig,ax = pls.subplots(1, 1,figsize=(8, 5))
                pls.scatter(df[feature], df[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
                ax.set_xlabel(feature)
                ax.set_ylabel(target)
                pls.show()

def one_to_many(df,fiterout=True):
    """Prints the maximum count of feature value thats corresponds to on value of onother feature.

    Args:
        df  (DataFrame): pandas DataFrame
        fiterout (bool): flag to find only one-to-one or one-to-many features

    """
    features =  df.columns.to_list()

    for f11 in  range(len(features)):
        for f22 in range(f11+1,len(features)):
            c2=df.groupby(features[f11])[features[f22]].nunique().max()
            c1=df.groupby(features[f22])[features[f11]].nunique().max()
            if not fiterout or c1<2 or c2<2:
                print(features[f11],' - ',features[f22],'\n\t relation ',c1,' : ',c2)

def tryanova( sub ,feature ,target ,maxcount ):
    try:
        #check variance
        if anova(sub[[feature,target]].dropna(),feature,target ,False):
            header(feature+' - '+ target,sz='h3')
            fig, ax = pls.subplots(figsize=(14 if maxcount>6 else 8, 8))
            sns.barplot(x=feature, y=target, data=sub, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            pls.show()
            turkey=turkeyHSD(sub,feature,target)
            if turkey.shape[0]:
                print(f'turkeyHSD\n{turkey}\n\n')
            return True

    except Exception as e:
        #pass
        print(e)

    return False

def is_depended(df,feature,target,cuts,maxcount):

    if not is_numeric(df[feature].dtype):
        topk=df[feature].value_counts().iloc[:maxcount].index
        return tryanova(df[df[feature].isin(topk)],feature,target,maxcount)

    if df[[target,feature]].corr().values[0,1]>0.1:
        return True
    if stats.spearmanr(df[feature], df[target]).correlation>0.1:
        return True
    if cuts:
        q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        df['cuts__'+feature]=pd.qcut(df[feature], q=q,duplicates='drop')
        topk=df['cuts__'+feature].value_counts().iloc[:maxcount].index
        return tryanova(df[df['cuts__'+feature].isin(topk)],'cuts__'+feature,target,maxcount)

    return False


def interactions2x(ddf,**kwarg):
    """2nd order interactions plots
        Check categorical varibles and binned numerical against the numerical 
        varibles by ANOVA and in case if diffrence in means is significant, 
        display plots and Tukey's HSD (honestly significant difference) test results

    Args:
        df    (DataFrame): pandas DataFrame
    Keyword Args:
        features  (list) : indepepended variables list
        target    (list) : continues depended variables list
        maxcount   (int) : maximum number of features to display
        cuts      (bool) : cut continues indepepended variables on bins or ignore

    Returns:
        depended features dictionary

    """
    df=ddf.copy()

    categorical= get_categorical(df,maxmissed=0.9,binary=True)
    numeric= [ c for c in df.columns if is_numeric(df[c].dtype)]
    both=set(numeric)|set(categorical)

    target=set(kwarg.get('target',numeric))&set(numeric)
    features=set(kwarg.get('features',both))&both

    if  len(features)<1 or len(target)<1:
        return None

    cuts=kwarg.get('cuts',False)
    maxcount=kwarg.get('maxcount',6)

    fanova={}
    for f  in features:
        depended=[]
        for t in target:
            if t==f:
                continue
            if is_depended(df,f,t,cuts,maxcount):
                depended.append(t)

        if len(depended)<1:
            continue

        print(f, ' - ',depended,'\n\n\n')
        fanova[f]= depended

        if f in numeric:
            if len(depended)>1:
                depended.append(f)
                corr=np.abs(df[depended].corr()[f])
                corr=corr.sort_values()[-maxcount:].index.to_list()
                sns.pairplot(df,vars=corr,corner=True)
                pls.show()
            else:
                sns.scatterplot(x=df[f],y=df[depended[0]])
                pls.show()
        elif len(depended)>1:
            if len(depended)>maxcount:
                depended=get_top_correlated(df[depended],maxcount=maxcount)
            topk=df[f].value_counts().iloc[:maxcount].index
            sns.pairplot(df[df[f].isin(topk)],vars=depended,hue=f,corner=True)
            pls.show()


    return fanova

def check_dependency3x(df,feature1,feature2,target,**kwarg):
    
    maxcount=kwarg.get('maxcount',6)
    verbose=kwarg.get('verbose',False)

    sub=df[df[feature1].isin(df[feature1].value_counts().iloc[:maxcount].index)]
    
    sub[feature1+feature2]=sub[feature1].astype(str)+sub[feature2].astype(str)
    top=sub[[feature1,feature2,feature1+feature2]].dropna()[feature2].value_counts().iloc[:maxcount].index
    sub=sub[sub[feature2].isin(top)]
    
    anova_res=False
    try:

        res=two_way_anova(sub.dropna(),feature1,feature2,target)
        header(target,sz='h5')
        if verbose:
            print(res)

        if res.iloc[2]['PR(>F)']<=0.05:
            anova_res=True
            print('Anova passed')
            print('turkeyHSD')
            print(turkeyHSD(sub,feature1+feature2,target))
        else:
            print('Anova faled to reject => no difference ')

    except Exception as e:
        print('nan',e)
        pass  
    
    if anova_res or verbose:
        print(feature1+'*'+feature2+'='+target)
        fig, ax = pls.subplots(figsize=(14, 8)  if maxcount>6 else (8, 6))
        sns.barplot(x=feature1, y=target, hue=feature2, data=sub,ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        pls.show()
        
    if anova_res: 
        sns.catplot(x=target, y=feature1, row=feature2, kind='box', orient='h', height=1.5, aspect=4,data=sub)
        pls.show()

    return anova_res
    
def interactions3x(df,**kwarg):
    """ 3nd order interactions plots.

    Check categorical varibles and binned numerical against the numerical varibles by
    two-way ANOVA   and in case if diffrence in means is significant, display plots

    Args:
        df  (DataFrame): pandas DataFrame
    Keyword Args:
        features  (list) : indepepended variables list
        targets   (list) : continues depended variables list
        maxcount   (int) : maximum number of features to display
        verbose   (bool) : detailed output or short
    Returns:
        depended features dictionary
    """    
    categorical= get_categorical(df,maxmissed=0.9,binary=True)
    numeric= [ c for c in df.columns if is_numeric(df[c].dtype)]
 
    targets=set(kwarg.get('targets',numeric))&set(numeric)
    features=kwarg.get('features',df.columns.to_list())

    categorical= set(categorical)&set(features)
    numeric= set(numeric)&set(features)

    if  len(features)<1 or len(targets)<1:
        return None
    
    fdf=df[features]
    if len(numeric):
        fdf=fdf.copy()
        for feature in numeric:
            if fdf[feature].nunique()>min( fdf[feature].shape[0]/2.0, 40):
                q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1]
                fdf[feature]=pd.qcut(fdf[feature], q=q, duplicates='drop').astype(str)   
            fdf[feature]=fdf[feature].astype(str)

    fanova={}    

    for i in range(len(features)-1):
        for j in range(i+1,len(features)):
            header(features[i]+' - '+ features[j],sz='h3')
            
            #if (features[i] in categorical) and (features[j] in categorical):
                #one-to-one
            if df.groupby(features[i])[features[j]].apply(lambda x: x.nunique() <2 ).all():
                print ('one-to-one skip')
                continue

            print('relation ',df.groupby(features[i])[features[j]].
                  nunique().max(),' : ',df.groupby(features[i])[features[j]].nunique().max())
            depended=[]    
            for target in targets:   
                if target==features[i] or target==features[j]:
                    continue
                if check_dependency3x(pd.concat([fdf[[features[i],features[j]]],df[target]],axis=1),
                                   features[i],features[j],target ,**kwarg):
                    depended.append(target)
                     
            if depended:
                print(depended)
                fanova[features[i]+' '+ features[j]]= depended.copy()

    return fanova

#=====================#=====================#=====================
# time series plots
#=====================#=====================#=====================

def plot_ntop_categorical_values(df,feature,target,**kwarg):
    """Plots n top feature values statistics per day

    Group dataframe by categorical feature value and calculates target statistics
    along datetime index

    Args:
        df  (DataFrame): pandas DataFrame
        feature   (str): feature for grouping (categorical)
        target    (str): feature to average
    Keyword Args:
        maxcount    (int): maximum number of features to display
        sample     (bool): if True select maxcount values randomly
                           if False select maxcount top features
        figsize   (tuple): plot size
        linewidth (float): line width
        period      (str): sampling period
        method_name (str): sum,count,mean etc

    """
    maxcount=kwarg.get('maxcount',4)
    if kwarg.get('sample',False):
        values=df[feature].value_counts().sample(maxcount).index.to_list()
    else:
        values=df[feature].value_counts()[:maxcount].index.to_list()

    if  len(values)==0:
        return

    figsize=kwarg.get('figsize',(20,4))
    linewidth=kwarg.get('linewidth',2.0)
    period=kwarg.get('period','1d')
    method_name=kwarg.get('method_name','count')

    resampler=df[df[feature]==values[0]][target].resample(period)
    ax=getattr(resampler,method_name)().plot(x_compat=True,figsize=figsize, grid=True,linewidth=linewidth)
    legend=[values[0]]

    for i in range(1,len(values)):
        resampler=df[df[feature]==values[i]][target].resample(period)
        getattr(resampler,method_name)().plot(x_compat=True,
                                              figsize=figsize,
                                              ax=ax,
                                              grid=True,
                                              linewidth=linewidth,
                                              title=f'{target} {method_name} per {feature} per day')
        legend.append(values[i])

    if len(values) > 1:
        ax.lines[1].set_linestyle(':')

    ax.lines[0].set_linestyle('--')
    pls.legend(legend, bbox_to_anchor=(1.2, 0))
    pls.show()


#=====================#=====================#=====================#=====================
# cramers V
#=====================#=====================#=====================#=====================
#Theil’s U, conditional_entropy (no symetrical)
#https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
#https://github.com/shakedzy/dython/blob/master/dython/nominal.py

def get_categorical(df,maxmissed=0.6,binary=True):
    """Select features for cramer v plot  - categorical or binary,
       features with too many different values are ignored
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','datetime64','m8[ns]']

    #keep columns with % of missed less then 60
    categoricals = df.loc[:, df.isnull().mean() <= maxmissed
                         ].select_dtypes(exclude=numerics).columns.to_list()

    if binary:
        #add binary columns
        bool_cols = [col for col in df.select_dtypes(include=numerics).columns.to_list() if
                   df[col].dropna().value_counts().index.isin([0,1]).all()]

        categoricals.extend(bool_cols)

    #drop columns with no variance and with too much variance (id etc)
    categoricals=[col for col in categoricals if
               df[col].dropna().nunique() >1 and df[col].nunique() < df.shape[0]/2]

    return categoricals

def cramers_corrected_stat(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def plot_cramer_v_corr(df,**kwarg):
    """Plots features correlation (Theil’s U, conditional_entropy) heatmap.

    Theil’s U, conditional_entropy (no symetrical)
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    https://github.com/shakedzy/dython/blob/master/dython/nominal.py
    https://stackoverflow.com/a/46498792/5863503

    Args:
        df  (DataFrame): pandas DataFrame
        target    (str): Dependend feature name

    Keyword Args:
        categoricals (list): list of feature to explore.
                             If is not set all availible categorical
                             or binary features are used
        maxcount   (int) :   maximum number of features to explore
        ignore     (list):   list of features to ignore
        figsize    (tuple):  plot size
        ax         (list):   matplotlib.axes, subplots list


    """
    ignore=kwarg.get('ignore',None)
    categoricals=kwarg.get('categoricals',get_categorical(df))
    maxcount=kwarg.get('maxcount',len(categoricals))
    if ignore is not None:
        categoricals=list(set(categoricals)-set(ignore))
    categoricals=categoricals[:maxcount]
    
    if len(categoricals)<=1:
        return

    figsize=kwarg.get('figsize',(10,10))
    ax=kwarg.get('ax',None)

    fig, ax = pls.subplots(1,1,figsize=figsize) if ax is None else None,ax

    correlation_matrix = pd.DataFrame(
        np.zeros((len(categoricals), len(categoricals))),
        index=categoricals,
        columns=categoricals
    )

    for col1, col2 in itertools.combinations(categoricals, 2):

        idx1, idx2 = categoricals.index(col1), categoricals.index(col2)
        correlation_matrix.iloc[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        correlation_matrix.iloc[idx2, idx1] = correlation_matrix.iloc[idx1, idx2]

    ax = sns.heatmap(correlation_matrix, annot=True, ax=ax)
    ax.set_title('Cramer V Correlation between Variables')

    if fig is not None:#local figure
        pls.show()


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

def header(title,sz='h2',interval='\n \n'):
    print(interval)
    display(HTML('<hr>'))
    display(HTML(f'<{sz} align=\"center\">{title}</{sz}>'))
    print('\n  ')

#=====================#=====================#=====================#=====================
# features
#=====================#=====================#=====================#=====================

def get_feature_type(s):
    if s.dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        return 'Numeric'
    elif is_time(s.dtype):
        return 'Time'
    elif s.dtype in [bool] or len(set(s.dropna().unique()) - set([False,True]))==0:
        return 'Boolean'
    else:
        return 'Categorical'

def get_feature_info(df,feature):
    if feature in df.columns:
        cardinality = df[feature].nunique()
        missed= 100 * df[feature].isnull().sum() / df.shape[0]
        feature_type = get_feature_type(df[feature])
        return feature_type,cardinality,missed
    else:
        return '','',''

def is_numeric(dtype):
    return dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]

def is_time(dtype):
    """Checks if dtype is one of time formats"""
    if '[ns]' in str(dtype) or 'datetime' in str(dtype) :
        return True
    return False

def safe_convert(s):
    """Tries to convert Series data to datetime if possible withought exception"""
    try:
        return pd.to_datetime(s, errors='ignore')
    except:
        pass
    return s
