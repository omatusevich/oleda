"""Exploratory Data Analysis with python.

   Automatic report generation from a pandas DataFrame.
   Insights from data.
   
   Typical usage example:
         import oleda
         oleda.report(df)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as pls

import shap
import seaborn as sns
from scipy import stats
import lightgbm as lgb
from IPython.display import display, HTML
from lightgbm import LGBMClassifier,LGBMRegressor

from .eda_anova import anova,two_way_anova,turkeyHSD

warnings.filterwarnings("ignore")
    
#=====================#=====================#=====================
# single dataset  eda
#=====================#=====================#=====================  

def report(df,target,**kwarg):  
    """Explore single dataset and create report .

       Report contains:
            missing values statistics
            information on each feature relevant for the feature type
            pearson correlation heatmap
            Cramers V staticstics
            pair plot for most correlated features
            if dataframe index is valid datetime index, time series plots are added to the report
       Dataset can be tested against an target variable (binary (0,1) or continues):
                oleda.report(df,target,ignore=[],nbrmax=20)
            in this case plots that show correlation with targed are added for each feature features 
            are sorterted according to their impotantce by shap nbrmax number of most important features 
            selected by shap to be explored features need to be ignored can be added in ignore list         
    Args:
        df       (object): Pandas dataframe        
    Keyword Args:
        dependency (bool): set to True to plot dependency plots 
        maxcount   (int) : maximum number of features to display
        ignore     (list): list of features to ignore         
    """

    ignore=kwarg.get('ignore',[])
    
    if target!=None and target not in df.columns.to_list():
        print("{} is not in dataframe - Ignore".format(target))
        target=None      
        
    #detect time columns
    df = df.apply(lambda col: safe_convert(col) if col.dtypes == object else col, axis=0)
    
    #print nans statistics
    header('Missed values' )
    print_na(df,**kwarg)
                      
    if (target is not None) and isNumeric(df[target].dtype) :
        header('Shap values')
        sorted_features=plot_shaps(df,target,**kwarg)
    else:
        sorted_features=[f for f in df.columns.tolist() if f not in ignore]
        #list(set(df.columns.tolist())-set(ignore))

    # if dataframe has timedate index - plot time series
    if target !=None and  isTime(df.index.dtype) and df.index.nunique()> 2:
        header('Time series' )
        ax=df[target].resample('1d').mean().plot( grid=True,x_compat=True,figsize=(20,4),linewidth=2.0,label=target)
        pls.title(' {} mean per day '.format(target))
        pls.legend()
        pls.show() 
 
    header('Features' )
    print_features(df,target,sorted_features)
     
    #for numeric variables only
    header('Pearson correlations' ) 
    plot_corr(df,**kwarg) 

    #correlations of categorical variables
    header('Cramers V staticstics' )    
    plot_cramer_v_corr(df,**kwarg) 

    #plot 10 most correlated features
    header('Top correlated features' )  
    f=get_top_correlated(df,th=0.099,maxcount=7)
    g=sns.pairplot(df[f])  
    
    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation = 90)
        ax.set_ylabel(ax.get_ylabel(), rotation = 0)
        ax.yaxis.get_label().set_horizontalalignment('right')
  
    
#=====================#=====================#=====================#
# shap 
#=====================#=====================#=====================#

def plot_shaps(df, target, **kwarg):
    """Calculates shapley values.

    Builds lightgbm model to predict target variable and calculates and plots 
    its shapley values.

    Args:
        df   (DataFrame): pandas DataFrame.
        target     (str): Dependend feature name
        
    Keyword Args:
        dependency (bool): set to True to plot dependency plots 
        maxcount   (int) : maximum number of features to display
        ignore     (list): list of features to ignore 
        
    Returns:
        Feature names sorted by importance 
    """    

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    maxcount  = kwarg.get('maxcount',df.shape[1])
    ignore    = kwarg.get('ignore',[])
    
    #doesn't work on time columns, remove id columns (all values are different), columns with all nulls     
    filterout=lambda f: (isTime(df[f].dtype) or df[f].isnull().values.all()\
                      or  (len(df[f].unique())>df.shape[0]/2.0 and str(df[f].dtype) not in numerics))

    features = [ f for f in df.columns if ~filterout(f) ]
    features.remove(target)
            
    features=list(set(features)-set(ignore))
    [print('Feature name {} cantains special JSON characters - Skip'.format(x))  for x in features  if ':' in x  ]
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
                            , 'min_sum_hessian_in_leaf' : 10
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
    shap.summary_plot(shap_values, df[features], max_display=maxcount, auto_size_plot=True)
    
    if binary_target:
        vals= np.abs(shap_values).mean(0)
    else:
        vals= shap_values
        
    feature_importance = pd.DataFrame(list(zip(df[features].columns, sum(vals))), 
                                      columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    sorted_features=feature_importance['col_name'].to_list()

    X=df.copy()

    if binary_target:
        shap.summary_plot(shap_values[1][:,:maxcount], df[features[:maxcount]])    
        
        if kwarg.get('dependency',True):
            
            for f in categorical_features:
                X[f]=  X[f].astype(object)
                X[f]=pd.factorize(X[f])[0]  

            for name in sorted_features[:maxcount]:
                #continue
                if name in categorical_features and df[name].astype(str).nunique()>100:
                    continue
                fig, ax = pls.subplots(1,1,figsize=(20,10))
                shap.dependence_plot(name, shap_values[1], X[features], 
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
    """Split continues variable on bins and plot per bins statistics.

    Split continues variable on bins and plot per bins target mean and count.
    If bins argument is absent , variable is splited on nbins 

    Args:
        df  (DataFrame): pandas DataFrame
        feature   (str): feature to split
        target    (str):  feature to calculate statiscs 

    Keyword Args:
        figsize (tuple): plot size 
        nbins     (int): number of bins 
        bins     (list): bins     
    """    
    figsize=kwarg.get('figsize',(8,4))
    bins=kwarg.get('bins',None)
    if bins==None:
        bins=np.arange(df[feature].min(),df[feature].max(),(df[feature].max()-df[feature].min())/(1+kwarg.get('nbins',10)))
    
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title('Histogram of {}'.format(feature)); 
    ax1.set_xlabel(feature)
    ax1.set_ylabel('count')
    ax2.set_xlabel(feature)
    ax2.set_ylabel(target)
    df.groupby(pd.cut(df[feature], bins=bins))[target].count().plot(kind='bar',ax=ax1,grid=True)
    df.groupby(pd.cut(df[feature], bins=bins))[target].mean().plot(kind='bar',ax=ax2,grid=True)  
    pls.show()  
    
def plot_qcuts(df,feature,target,**kwarg):
    """Split continues variable on quantiles and plot per quantiles statistics.

    Split continues variable on quantiles and plot per quantiles target mean and count.
    If quantiles argument is absent , variable split on 10 quantiles 

    Args:
        df  (DataFrame): pandas DataFrame
        feature   (str): feature to split
        target    (str):  feature to calculate statiscs 

    Keyword Args:
        figsize (tuple) : plot size 
        qbins   (list)  : quantiles 
    """
    figsize=kwarg.get('figsize',(8,4))
    qbins=kwarg.get('qbins',None)
    
    if qbins==None:
        qbins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title('Histogram of {}'.format(feature)); 
    ax1.set_xlabel(feature) 
    ax1.set_ylabel('count')
    ax2.set_xlabel(feature) 
    ax2.set_ylabel(target)
    df.groupby(pd.qcut(df[feature], q=qbins,duplicates='drop'))[target].count().plot(kind='bar',ax=ax1,grid=True)
    df.groupby(pd.qcut(df[feature], q=qbins,duplicates='drop'))[target].mean( ).plot(kind='bar',ax=ax2,grid=True)

    pls.show()    
                      
#=====================#=====================#=====================#=====================
# categorical 
#=====================#=====================#=====================#=====================

def plot_stats(df,feature,target,**kwarg):
    """Plots target statistcs per feature value.

    Plots target mean and count per categorical feature value.

    Args:
        df  (DataFrame): pandas DataFrame
        feature   (str): feature to group data
        target    (str): feature to calculate statiscs 
          
    Keyword Args:
        ax         (list): matplotlib.axes, subplots list 
        maxcount   (int) : maximum number of features to display
        figsize    (list): figure size
        sortby     (str) : sort column name
    """    
    maxcount=kwarg.get('maxcount',20)
    ax=kwarg.get('ax',None)
    sort=kwarg.get('sortby','Count ')
    end=maxcount

    cat_count = df[feature].value_counts().reset_index()
    cat_count.columns = [feature,'Count ']
    cat_count.sort_values(by=sort, ascending=False, inplace=True)

    cat_perc = df[[feature, target]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_count,on=feature)
    cat_perc.sort_values(by=sort, ascending=False, inplace=True)
    
    size=kwarg.get('figsize',(12,6) if len(cat_count[:maxcount]) <=40 else (12,14))
    if ax is None:
        fig, ax = pls.subplots(ncols=2, figsize=size)
        
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax[0], x = feature, y="Count ",order=cat_count[feature][:maxcount],data=cat_count[:maxcount])
    s.set_xticklabels(s.get_xticklabels(),rotation=90)   
    
    s = sns.barplot(ax=ax[1], x = feature, y=target, order=cat_perc[feature][:maxcount], data=cat_perc[:maxcount])  
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
        
    pls.ylabel(target, fontsize=10)
    pls.tick_params(axis='both', which='major', labelsize=10)

    if kwarg.get('showfigure',True):
        pls.show()
        
def plot_melt(df,feature,target1,target2,**kwarg):
    """Plots two targets statistcs per feature value.

    Plots two targets mean and count per categorical feature value.

    Args:
        df  (DataFrame): pandas DataFrame
        feature   (str):  feature to group data
        target1   (str):  feature to calculate statiscs 
        target2   (str):  feature to calculate statiscs 
        
    Keyword Args:
        maxcount   (int) : maximum number of features to display       
    """    
    end=kwarg.get('maxcount',20)
    cat_count = df[feature].value_counts().reset_index()
    cat_count.columns =[feature,'Count ']
    cat_count.sort_values(by='Count ', ascending=False, inplace=True)

    cat_perc = df[[feature, target1]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_count,on=feature)

    cat_perc2 = df[[feature, target2]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_perc2,on=feature)   
    cat_perc.sort_values(by='Count ', ascending=False, inplace=True)
    cat_perc=cat_perc[:end]
    
    data_melted = pd.melt(cat_perc[[feature,target1,target2]], id_vars=feature,\
                           var_name="source", value_name="value_numbers")
    
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=(12,6))
    sns.set_color_codes("pastel") 
    s = sns.barplot(ax=ax1, x = feature, y="Count ",order=cat_count[feature][:end],data=(cat_count[:end]))
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
  
    s = sns.barplot(ax=ax2, x = feature, y="value_numbers",hue="source", order=data_melted[feature][:min(end,cat_count.shape[0])],data=(data_melted))
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    pls.tick_params(axis='both', which='major', labelsize=10)
    pls.legend(bbox_to_anchor=(2, 0))
    pls.show();
    
#==========================================#=====================#=====================
# nan
#==========================================#=====================#=====================

def plot_na(df,** kwarg):
    """Plots persent of nans in each data frame column.

    Args:
        df  (DataFrame): pandas DataFrame
        
    Keyword Args:
        figsize (tuple): plot size        
    """    
    #pls.style.use('seaborn-talk')

    figsize=kwarg.get('figsize',(18,6))
    fig = pls.figure(figsize=figsize)
    miss = pd.DataFrame((df.isnull().sum())*100/df.shape[0]).reset_index()
    ax = sns.pointplot(x="index",y=0,data=miss)
    pls.xticks(rotation =90,fontsize =7)
    pls.title("Missed data")
    pls.ylabel(" %")
    pls.xlabel("Features")

def print_na(df, **kwarg ):
    """Print persent of nans in each column.

    Args:
        df  (DataFrame): pandas DataFrame
        
    Keyword Args:
        maxcount   (int) : maximum number of features to display
        
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
        A list of most correlated feature names .

    """    
    numeric= [ c for c in df.columns if df[c].dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]]
    corr=df[numeric].corr()
    if corr.shape[0]>0:
        cx=corr.unstack().dropna().abs().sort_values(ascending=False).reset_index()
        return list(cx[(cx[0]>th)&(cx.level_0!=cx.level_1)]['level_0'].drop_duplicates(keep='first')[:maxcount].values)   
    else:
        return []
      
def plot_corr(df, **kwargs):   
    """Plots correlation heatmap for all numerical features.

    Find all numerical features and and plots their correlation heatmap.

    Args:
        df  (DataFrame): pandas DataFrame
          
    Keyword Args:
        maxcount     (int): maximum number of features to draw 
        figsize    (tuple): heatmap size 
        features    (list): features list to investigate
        features_ext(list): to calculate correlation between two different sets of features
       
    """
    numeric= [ c for c in df.columns if df[c].dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]]
    features=kwargs.get('features',numeric)
    features_ext=kwargs.get('features_ext',features)      
    
    maxcount=kwargs.get('maxcount',len(numeric))
    
    corr=df[numeric].corr()
    cx=corr.abs().unstack().sort_values(ascending=False).reset_index().dropna() 
    if 'features' not in kwargs.keys():
        features=list(cx[cx.level_0!=cx.level_1]['level_0'].drop_duplicates(keep='first')[:maxcount].values)
        features_ext=features

    if (len(features)+len(features_ext))<2:
        return
    
    v=min(len(features),min(maxcount,20))
    figsize=kwargs.get('figsize',(v,v))

    fig, ax = pls.subplots(1,1,figsize=figsize)
    sns.heatmap( corr[corr.index.isin(features)][features_ext], cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    pls.title('Correlation Heatmap')
    pls.show() 
    return 
   
       
#=====================#=====================#=====================#=====================
# report
#=====================#=====================#=====================#=====================

def print_features(df,target=None,sorted_features=[]):   
    """Plots feature statistics.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        df  (DataFrame): pandas DataFrame

    """
    #explore features selected by shap (sorted_features)
    features = sorted_features if len(sorted_features)>0 else df.columns.to_list()
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    numeric=list(set(numeric)&set(sorted_features))
    
    for feature in features:
                                                                   
        if feature==target:
            continue
                                                                   
        print('\n ')
        display(HTML("<hr>"))
        display(HTML("<h3 align=\"center\">{}</h3>".format(feature)))
        print('\n ') 
                                                                   
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
             explore_numerical(df,feature,target)                           
        else:
             explore_timecolumn(df,feature,target)
                
def explore_timecolumn(df,feature,target): 
        info = pd.DataFrame(index=['dType :' ,'Min :', 'Max :'],columns=[' '])       
        info[' ']=[df[feature].dtype,df[feature].min(),df[feature].max()]
        print(info.head())
        print('\n ')            
        print("Time column skip plotting ")  
        
def explore_categorical(df,feature,target,numeric,maxcount=4):
   
    df[feature]=df[feature].fillna('Missed value')

    tg_cardinality=df[target].nunique() 
    cardinality=df[feature].nunique()
    
    if cardinality > df.shape[0]/2.0 :
        print("Too many values to plot ")
    elif df[feature].isnull().values.all() :
        print("All values are null")
    elif cardinality<2:
        print("Zero variance")
                                                                   
    else:
        if target != None :
            plot_stats(df,feature,target,maxcount=30)
        elif cardinality<=40:
            fig,ax =  pls.subplots(1, 1,figsize=(9, 5))
            df[feature].astype('str').hist()
            pls.xticks(rotation='vertical')
            pls.show()
        else:
            fig,ax =  pls.subplots(1, 1,figsize=(9, 5))
            f=df[feature].value_counts()[:40].index
            df[df[feature].isin(f)][feature].hist()
            pls.xticks(rotation='vertical')
            pls.show()

        #count of records with feature=value per day
        if target != None and  isTime(df.index.dtype) and df.index.nunique()>2:
            display(HTML("<h3 align=\"center\">Top {} count per day</h3>".format(feature)))
            plot_ntop_categorical_values(df,feature,target,method_name='count')

            #mean of target for records with feature=value per day
            display(HTML("<h3 align=\"center\">{} mean per day </h3>".format(target)))
            plot_ntop_categorical_values(df,feature,target,method_name='mean')

        if target != None and tg_cardinality>2 and cardinality<40 :
            if tg_cardinality>15:
                sns.catplot(y=feature,x=target,data=df, orient="h", kind="box",height=7)
            else:
                sns.catplot(y=feature,x=target,data=df, orient="h", kind="box")
            pls.show()
            
    #partitioning
    if len(numeric)>1:
        depended =[]

        #to speed up - select 4 most popopular values
        top=df[feature].value_counts().iloc[:maxcount].index
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
                                
def explore_numerical(df,feature,target):
   
        tg_cardinality=df[target].nunique() 
        cardinality=df[feature].nunique()
    
        info = pd.DataFrame(
        index=['dType :' ,'Min :', 'Max :', 'Mean :', 'Std :'],
        columns=[' '])       
        info[' ']=[df[feature].dtype,df[feature].min(),df[feature].max(),df[feature].mean(),df[feature].std() ]
        print(info.head())
        print('\n ')
        
        #pairwise_feature_sum_per_day(df1,df2,feature)
        #pairwise_feature_mean_per_day(df1,df2,feature)
        if cardinality<=40:
            if target !=None :
                plot_stats(df,feature,target,maxcount=40)
            else:
                pls.hist(df[feature])
                pls.show()

            if target !=None and tg_cardinality>2 and cardinality>2 :
                
                fig,ax =  pls.subplots(1, 1,figsize=(8, 5))
                pls.scatter(df[feature], df[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
                ax.set_xlabel(feature)
                ax.set_ylabel(target)
                pls.show()

            if target !=None and tg_cardinality>2:
                if tg_cardinality>15:
                    sns.catplot(y=feature,x=target,data=df, orient="h", kind="box",height=7)
                else:
                    sns.catplot(y=feature,x=target,data=df, orient="h", kind="box")
                pls.show()
        else:
            #df[feature].hist() 

            #distribution
            fig,ax = pls.subplots(1, 2,figsize=(16, 5))
            sns.distplot(df[feature],kde=True,ax=ax[0]) 
            ax[0].axvline(df[feature].mean(),color = "k",linestyle="dashed",label="MEAN")
            ax[0].legend(loc="upper right")
            ax[0].set_title('Skewness = {:.4f}'.format(df[feature].skew()))
            sns.boxplot(df[feature],color='blue',orient='h',ax=ax[1])
            pls.show()
                
            if  target !=None :

                if tg_cardinality < 10:
                    fig,ax = pls.subplots(1, 2,figsize=(16, 5))
                    #ax[0].scatter(df[feature], df[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
                    #ax[0].set_xlabel(feature)
                    #ax[0].set_ylabel(target)
                    sns.kdeplot(data=df, x=feature, hue=target, multiple="stack" ,ax=ax[0])
                    sns.violinplot(x=target, y=feature, data=df, ax=ax[1], inner='quartile')
                    pls.show()  
                else:
                    #continues - continues
                    plot_qcuts(df,feature,target)
                    q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
                    df['cuts__'+feature]=pd.qcut(df[feature],q=q,duplicates='drop')
                    sns.catplot(x=target,y='cuts__'+feature,data=df, orient="h", kind="box") 
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
        fiterout (bool): flag to finsd only one-to-one or one-to-many features

    """    
    features =  df.columns.to_list()
    
    for f11 in  range(len(features)):
        for f22 in range(f11+1,len(features)):
            c2=df.groupby(features[f11])[features[f22]].nunique().max()
            c1=df.groupby(features[f22])[features[f11]].nunique().max()
            if not fiterout or c1<2 or c2<2:
                print(features[f11],' - ',features[f22],'\n\t relation ',c1,' : ',c2)

def tryanova(sub,f,t,maxcount):
    try:
        #check variance 
        if anova(sub,f, t,False):
            header(f+' - '+ t,sz='h3')
            
            if maxcount>6:
                fig, ax = pls.subplots(figsize=(14, 8))
            else:
                fig, ax = pls.subplots(figsize=(8, 6))    

            sns.barplot(x=f, y=t, data=sub, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            pls.show()
            turkey=turkeyHSD(sub,f,t)
            if turkey.shape[0]>0:
                print('turkeyHSD')
                print(turkey)
                print('\n')
            return True
    except Exception as e:
        #pass
        print(e)
    return False

def interactions2x(ddf,**kwarg):
    """2 nd order interactions plots
    
         Check categorical varibles and binned numerical against the numerical varibles by ANOVA 
         and in case if diffrence in means is significant, display plots and Tukey's HSD
         (honestly significant difference) test results

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
    numeric= [ c for c in df.columns if df[c].dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]]

    target=kwarg.get('target',None)
    features=kwarg.get('features',None)
    cuts=kwarg.get('cuts',False)
    maxcount=kwarg.get('maxcount',6)
    
    with pd.option_context('mode.use_inf_as_na', True):    
        if df.isnull().values.any():
            print('Warning: dataframe contains nan or inf , please fix or drop them to obtain better results. \n')
            df[numeric]=df[numeric].fillna(0)
            df[categorical]=df[categorical].fillna('missed')
    
    
    target=set(numeric)&set(target) if target is not None else set(numeric)        
    features=(set(numeric)|set(categorical))&set(features) if features is not None else (set(numeric)|set(categorical))
   
    if  len(features)<1 or len(target)<1:
        return None 
    
    fanova={}
    for f in features & set(categorical):
        topK=df[f].value_counts().iloc[:maxcount].index
        sub=df[df[f].isin(topK)].copy()
        sub['count']=1
        depended=[]
        for t in target:
            if t==f:
                continue
            if tryanova(sub,f,t,maxcount):
                depended.append(t)
        if len(depended)>0:
                print(f, ' - ',depended,'\n\n\n') 
                fanova[f]= depended   
                
        if len(depended)>1:
                if len(depended)>maxcount:
                    depended=get_top_correlated(sub[depended],maxcount=maxcount)
                sns.pairplot(sub,vars=depended,hue=f,corner=True)
                pls.show()        
    if cuts:            
        for f in features & set(numeric):
            depended=[]
            for t in target:
                if t==f:
                    continue
                if df[[t,f]].corr().values[0,1]>0.1:
                    depended.append(t)
                elif stats.spearmanr(df[f], df[t]).correlation>0.1:
                    depended.append(t)
                elif cuts:
                    q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    df['cuts__'+f]=pd.qcut(df[f], q=q,duplicates='drop')
                    topK=df['cuts__'+f].value_counts().iloc[:maxcount].index
                    sub=df[df['cuts__'+f].isin(topK)].copy()
                    sub['count']=1
                    if tryanova(sub,'cuts__'+f,t,maxcount):
                        depended.append(t)
            if depended:
                    print(f, ' - ',depended,'\n\n\n') 
                    fanova[f]= depended.copy()

            if len(depended)>1:  
                    depended.append(f)
                    corr=df[depended].corr()[f]
                    corr=np.abs(corr)
                    corr=corr.sort_values()[-maxcount:].index.to_list()                
                    sns.pairplot(df,vars=corr,corner=True)
                    pls.show()
            elif len(depended)>0: 
                sns.scatterplot(x=df[f],y=df[depended[0]])
                pls.show()
                
    return fanova


def interactions3x(ddf,features=None,**kwarg):   
    """ 3nd order interactions plots.

    Check categorical varibles and binned numerical against the numerical varibles by 
    two-way ANOVA   and in case if diffrence in means is significant, display plots
    
    Args:
        df  (DataFrame): pandas DataFrame
    Keyword Args:
        features  (list) : indepepended variables list
        target    (list) : continues depended variables list
        maxcount   (int) : maximum number of features to display
        verbose   (bool) : detailed output or short
    Returns:
        depended features dictionary

    """    
    df=ddf.copy()
    categorical= get_categorical(df,maxmissed=0.9,binary=True)
    numeric= [ c for c in df.columns if df[c].dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]]
    
    target=kwarg.get('target',None)
    features=kwarg.get('features',None)
    verbose=kwarg.get('verbose',False)
    maxcount=kwarg.get('maxcount',6)
    
    with pd.option_context('mode.use_inf_as_na', True):    
        if df.isnull().values.any():
            print('Warning: dataframe contains nan or inf , please fix or drop them to obtain better results. \n')
            df[numeric]=df[numeric].fillna(0)
            df[categorical]=df[categorical].fillna('missed')

    
    target=set(numeric)&set(target) if target is not None else set(numeric)        
  
    if features is not None:
        categorical= set(categorical)&set(features)
        numeric= set(numeric)&set(features)
        
    features=list(categorical)
    fanova={}
    
    for f in numeric:  
        cardinality=df[f].nunique()                                                
        if cardinality < df.shape[0]/2.0 and cardinality>=2 and cardinality < 40:
            features.append(f)
            continue
        else:
            q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1]
            df['cuts__'+f]=pd.qcut(df[f], q=q,duplicates='drop')
            df['cuts__'+f]=df['cuts__'+f].astype(str)
            features.append('cuts__'+f)
                
    if  len(features)<1 or len(target)<1:
        return None 
    
    for i in range(len(features)-1):
        f1=features[i]
        for j in range(i+1,len(features)):
            sub1=df[df[f1].isin(df[f1].value_counts().iloc[:maxcount].index)]
            f2=features[j]
            header(f1+' - '+ f2,sz='h3')
            
            #one-to-one
            if df.groupby(f1)[f2].apply(lambda x: x.nunique() <2 ).all():
                print ('one-to-one skip')
                continue
                
            print('relation ',df.groupby(f1)[f2].nunique().max(),' : ',df.groupby(f2)[f1].nunique().max())
            sub1[f1+f2]=sub1[f1].astype(str)+sub1[f2].astype(str)
            
            top=sub1[[f1,f2,f1+f2]].dropna()[f2].value_counts().iloc[:maxcount].index
            sub=sub1[sub1[f2].isin(top)]
            
            depended=[] 
            
            for t in target:
                
                if ('cuts__'+t) in [f1, f2] or t in [f1,f2]:
                    continue
                    
                print('\n\n',t)
                
                try:
                                        
                    res=two_way_anova(sub[[f1,f2,f1+f2,t]].dropna(),f1,f2,t)
                    if verbose:
                        print(res)

                    if(res.iloc[2]['PR(>F)']<=0.05):
                        print('Anova passed')
                        #print(f1,f2,t)
                        depended.append(t)
                        if maxcount>6:
                            fig, ax = pls.subplots(figsize=(14, 8))
                        else:
                            fig, ax = pls.subplots(figsize=(8, 6)) 
                            
                        sns.barplot(x=f1, y=t, hue=f2, data=sub,ax=ax)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                        pls.show()
                        
                        print('turkeyHSD')
                        print(turkeyHSD(sub,f1+f2,t))
                    else:
                        if verbose:
                            if maxcount>6:
                                fig, ax = pls.subplots(figsize=(14, 8))
                            else:
                                fig, ax = pls.subplots(figsize=(8, 6)) 
                                
                            bar=sns.barplot(x=f1, y=t, hue=f2, data=sub,ax=ax)
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                            pls.show()                            
                        print('Anova faled to reject => no difference ')

                except Exception as e:
                    print('nan',e)
                    pass
                
            if depended:        
                print(depended)
                fanova[f1+' '+ f2]= depended.copy()
            
            if depended:         
                if len(depended)>1: 
                    top=sub[[f1,f2,f1+f2]].dropna()[f1+f2].value_counts().index[:maxcount]
                    sns.pairplot(sub[sub[f1+f2].isin(top)],vars=depended,hue=f1+f2,corner=True)
                    pls.show()
                for t in depended:
                    print(f1+'*'+f2+'='+t)
                    g = sns.catplot(x=t, y=f1, row=f2, kind="box", orient="h", height=1.5, aspect=4,data=sub)
                    pls.show()
                    
    return fanova

#=====================#=====================#=====================
# time series plots
#=====================#=====================#=====================

def plot_ntop_categorical_values(df,feature,target,**kwarg):
    """Plots n top feature values counts per day   

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
    period=kwarg.get('period',"1d")
    method_name=kwarg.get('method_name','count')
    
    resampler=df[df[feature]==values[0]][target].resample(period)
    ax=getattr(resampler,method_name)().plot(x_compat=True,figsize=figsize, grid=True,linewidth=linewidth)
    legend=[values[0]]
    
    for i in range(1,len(values)):
        resampler=df[df[feature]==values[i]][target].resample(period)
        getattr(resampler,method_name)().plot(x_compat=True, figsize=figsize,ax=ax, grid=True, linewidth=linewidth,
                                              title='{} {} per day'.format(feature,method_name))
        legend.append(values[i])

    if len(values) > 1:
        ax.lines[1].set_linestyle(":")
    ax.lines[0].set_linestyle("--")
    pls.legend(legend, bbox_to_anchor=(1.2, 0))
    pls.show()
    
    
#=====================#=====================#=====================#=====================
# cramers V
#=====================#=====================#=====================#=====================
#Theil’s U, conditional_entropy (no symetrical)
#https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
#https://github.com/shakedzy/dython/blob/master/dython/nominal.py
   
import scipy.stats as ss
import itertools

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

def cramers_corrected_stat(confusion_matrix):
    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
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
    
    fig=None
    if ax==None:
        fig, ax = pls.subplots(1,1,figsize=figsize)
        
    correlation_matrix = pd.DataFrame(
        np.zeros((len(categoricals), len(categoricals))),
        index=categoricals,
        columns=categoricals
    )

    for col1, col2 in itertools.combinations(categoricals, 2):

        idx1, idx2 = categoricals.index(col1), categoricals.index(col2)
        correlation_matrix.iloc[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        correlation_matrix.iloc[idx2, idx1] = correlation_matrix.iloc[idx1, idx2]

    ax = sns.heatmap(correlation_matrix, annot=True, ax=ax); 
    ax.set_title("Cramer V Correlation between Variables");

    if fig!=None:#local figure
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

def header(title,sz='h2'):
    print('\n \n')
    display(HTML("<hr>"))
    display(HTML("<{} align=\"center\">{}</{}>".format(sz,title,sz)))
    print('\n  ') 
    
#=====================#=====================#=====================#=====================
# features 
#=====================#=====================#=====================#=====================  
        
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
        return feature_type,cardinality,missed
    else:
        return "","",""   

def isNumeric(dtype):
    return dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]

def isTime(dtype):
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