"""Datasets comparision.

This module provides functions for comparing two datasets and identifying
the features with the most significant differences in their distributions.
It can be utilized for detecting data drift or comparing clusters.

Typical usage example:

  pairwise_report(dataset1,dataset2)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pls
import seaborn as sns
from IPython.display import display, HTML

from .eda import plot_stats
from .eda import plot_cramer_v_corr,safe_convert,header
from .eda import plot_ntop_categorical_values
from .eda import plot_shaps,get_feature_info
from .eda import missing_values_table

#=====================#=====================#=====================
# dataset comparison  pairwise report
#=====================#=====================#=====================

def pairwise_report(df1,df2,**kwarg):
    """Dataset comparison.

    Compare two datasets and creates html report.

    Args:
        df1  (DataFrame): first  pandas DataFrame to compare
        df2  (DataFrame): second pandas DataFrame to compare
    Keyword Args:
        maxcount    (int): Maximum number of features to display.
        ignore     (list): List of features to ignore
        target   (string): Target variable for both datasets
        figsize   (tuple): Size of the plot.
        linewidth (float): Line width
        period      (str): Sampling period for time series
        maxshap     (str): Maximum number of features for which shapley values are displayed
        full       (bool): If True extended version of report is created
.
    """
    target=kwarg.get('target',None)
    maxshap=kwarg.get('maxshap',max(df1.shape[1],df2.shape[1]))
    kwarg['maxshap']=maxshap
    maxcount=kwarg.get('maxcount',min(20,maxshap))

    if target is not None and target not in df1.columns.to_list():
        print(f'{target} not in dataframe - Ignore')
        target=None

    #detect time columns
    df1 = df1.apply(lambda col: safe_convert(col) if col.dtypes == object else col, axis=0)
    df2 = df2.apply(lambda col: safe_convert(col) if col.dtypes == object else col, axis=0)

    #print shap values for each frame predicting targed
    header('Shap values' )

    if target is None:#compare datasets
        sorted_features=plot_shap_pairwise(df1,df2,**kwarg)
    else:           #compare acconding to target
        sorted_features1=plot_shap(df1,target,**kwarg)
        sorted_features2=plot_shap(df2,target,**kwarg)
        sorted_features=sorted_features1 if len(sorted_features1)>len(sorted_features2) else sorted_features2

    sorted_features=sorted_features[:maxshap]

    # if dataframe has timedate index - plot time series
    if (target is not None and df1.index.dtype==np.dtype('datetime64[ns]')
                     and df2.index.dtype==np.dtype('datetime64[ns]')):
        header('Time series')
        #print frames records counts per day side by side
        pairwise_target_stat_per_day(df1,df2,target,method_name='count')

    #print each feature stat
    header('Features info')
    print_features_pairwise(df1,df2,target=target,sorted_features=sorted_features,**kwarg)

    if kwarg.get('full',True):

        figsize=kwarg.get('figsize',(6,6))
        figsize2x1=(figsize[0]*2,figsize[1])

        header('Missed values')
        plot_na_pairwise(df1,df2,figsize2x1)
        print('\n \n ')
        #print columns with % of missing values
        print_na_pairwise(df1,df2)

        #for numeric variables only
        header('Pearson correlations')
        plot_correlations_pairwise(df1,df2,maxcount)

        #correlations of categorical variables
        header('Cramers V staticstics')
        plot_cramer_v_corr_pairwise(df1,df2,maxcount)

def _create_info_frame(df,feature):
    feature_type,cardinality,missed = get_feature_info(df,feature)
    info = pd.DataFrame(index=['Type :' ,'Distinct count :', 'Missed %:'], columns=[' '])
    info[' '] =[feature_type,cardinality,missed]
    if feature_type == 'Numeric':
        info.loc['Median :',' ']=df[feature].median()
        info.loc['Mean :',' ']=df[feature].mean()
        info.loc['dType :',' ']=df[feature].dtype
        info.loc['Min :',' ']=df[feature].min()
        info.loc['Max :',' ']=df[feature].max()
        info.loc['Std :',' ']=df[feature].std()
    return info

def print_features_pairwise(df1,df2,target=None,sorted_features=None,**kwarg):
    """Prints feature statics side by side for dataset comparision
    """
    figsize=kwarg.get('figsize',(6,6))
    figsize2x2=(figsize[0]*2,figsize[1]*2)
    figsize2x1=(figsize[0]*2,figsize[1])
    maxcount=kwarg.get('maxcount',20)

    features = sorted_features if sorted_features is not None else list(set(df1.columns.to_list())
                                                                       &set(df2.columns.to_list()))
    for feature in features:
        if feature==target:
            continue
        print('\n ')
        header(feature,sz='h3')
        display_side_by_side([_create_info_frame(df1,feature),
                              _create_info_frame(df2,feature)],['Frame 1','Frame 2'])
        print('\n ')
        feature_type,cardinality,_ = get_feature_info(df1,feature)
        cardinality2=df2[feature].nunique()
        noempty1=cardinality>0
        noempty2=cardinality2
        cardinality=max(cardinality,cardinality2)

        if (feature_type =='Categorical' or feature_type=='Boolean' or
            cardinality <= maxcount ):
            if  cardinality>df1.shape[0]/2.0 :
                print('Too many values to plot ')
            elif df1[feature].isnull().values.all() and df2[feature].isnull().values.all():
                print('All values are null')
            elif cardinality<2:
                print('Zero variance')
            else:
                if target is None:
                    plot_pairwise_barplot(df1,df2,feature,maxcount=maxcount,figsize=figsize2x1)
                else:
                    plot_pairwise_stats(df1,df2,feature,target,maxcount=maxcount,figsize=figsize2x2)
                    #count of records with feature=value per day
                    if df1.index.dtype==np.dtype('datetime64[ns]') and df2.index.dtype==np.dtype('datetime64[ns]'):
                        display(HTML(f'<h3 align=\'center\'>Top {feature} count per day</h3>'))
                        if noempty1: plot_ntop_categorical_values(df1,feature,target,maxcount=4,method_name='count')
                        if noempty2: plot_ntop_categorical_values(df2,feature,target,maxcount=4,method_name='count')

                        #mean of target for records with feature=value per day
                        display(HTML(f'<h3 align=\'center\'>{target} mean per day </h3>'))
                        if noempty1: plot_ntop_categorical_values(df1,feature,target,maxcount=4,method_name='mean')
                        if noempty2: plot_ntop_categorical_values(df2,feature,target,maxcount=4,method_name='mean')

        elif feature_type =='Numeric':
            pairwise_density_plot(df1,df2,feature,figsize=figsize2x1)

            #fig,(ax1, ax2) = pls.subplots(ncols=2,figsize=figsize2x1)#(16, 5))
            #if noempty1: sns.boxplot(df1.reset_index()[feature],color='blue',orient='h',ax=ax1)
            #if noempty2: sns.boxplot(df2.reset_index()[feature],color='blue',orient='h',ax=ax2)
            #pls.show()

            if  target is not None:
                pairwise_scatter_plot(df1,df2,feature,target)
                fig,(ax1, ax2) = pls.subplots(ncols=2,figsize=figsize2x1)
                if noempty1: sns.violinplot(x=target, y=feature, data=df1, ax=ax1, inner='quartile')
                if noempty2: sns.violinplot(x=target, y=feature, data=df2, ax=ax2, inner='quartile')
                pls.show()

        else:
            print('Time column - skip  ')

def plot_shap_pairwise(df1, df2,**kwarg):
    """
    Add an indicator variable to each dataset to distinguish them, and then concatenate both datasets.
    The resulting combined dataset is then passed into a function to calculate Shap values,
    in order to identify the most significant features for predicting this indicator variable.
    """
    target='fake_target_547454889'
    x=df1.copy()
    x[target]=0
    xx=df2.copy()
    xx[target]=1
    x=pd.concat([x,xx],axis=0)

    return plot_shaps(x,target,**kwarg)

def plot_cramer_v_corr_pairwise(df1,df2,maxcount=20,figsize=(40,20)):
    fig, ax = pls.subplots(nrows=1,ncols=2,figsize=figsize)
    plot_cramer_v_corr(df1,maxcount=maxcount,ax=ax[0])
    plot_cramer_v_corr(df2,maxcount=maxcount,ax=ax[1])
    pls.show()

def plot_correlations_pairwise(df1,df2,maxcount=20,figsize=(40,20)):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    l1=df1.select_dtypes(include=numerics).columns.to_list()
    l2=df2.select_dtypes(include=numerics).columns.to_list()
    if len(l1)>=2 or len(l2)>=2:
        fig,(ax1, ax2) = pls.subplots(ncols=2,figsize=figsize)
        sns.heatmap(df1[l1[:maxcount]].corr(), cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6,ax=ax1)
        ax1.set_title('Correlation Heatmap')
        sns.heatmap(df2[l2[:maxcount]].corr(), cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6,ax=ax2)
        ax2.set_title('Correlation Heatmap')
        pls.show()
    else:
        print('Less then 2 numeric columns - skip ')

def print_na_pairwise(df1,df2,max_row=60):
    display_side_by_side([missing_values_table(df1).head(max_row),
                          missing_values_table(df2).head(max_row)],
                         ['Frame 1 ','Frame 2'])

def plot_na_pairwise(df1,df2,figsize=(18,6)):
    pls.style.use('seaborn-talk')
    fig = pls.figure(figsize=figsize)
    miss_1 = pd.DataFrame((df1.isnull().sum())*100/df1.shape[0]).reset_index()
    miss_2 = pd.DataFrame((df2.isnull().sum())*100/df2.shape[0]).reset_index()
    miss_1['type'] = 'first'
    miss_2['type'] = 'second'
    missing = pd.concat([miss_1,miss_2],axis=0)
    sns.pointplot(x='index',y=0,data=missing,hue='type')
    pls.xticks(rotation =90,fontsize =7)
    pls.title('Missed data')
    pls.ylabel(' %')
    pls.xlabel('Features')
    pls.show()

def pairwise_density_plot(df1,df2,feature, figsize=(12,6)):
    fig,(ax1, ax2) = pls.subplots(ncols=2,figsize=figsize)
    if  not df1[feature].isnull().values.all():
        sns.histplot(df1[feature], stat='density',ax=ax1,kde=True)
    ax1.axvline(df1[feature].mean(),color = 'k',linestyle='dashed',label='MEAN')
    ax1.legend(loc='upper right')
    ax1.set_title('{} Skewness = {:.4f}'.format('Frame 1',df1[feature].skew()))

    if  not df2[feature].isnull().values.all():
        sns.histplot(df2[feature], stat='density',ax=ax2,kde=True)
    ax2.axvline(df2[feature].mean(),color = 'k',linestyle='dashed',label='MEAN')
    ax2.legend(loc='upper right')
    ax2.set_title('{} Skewness = {:.4f}'.format('Frame 2',df2[feature].skew()))
    pls.show()

def pairwise_hist_plot(df1,df2,feature,bins=30, figsize=(12,6)):
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    if  not df1[feature].isnull().values.all():
        ax1.hist(df1[feature] , edgecolor = 'k', bins = bins)
    if  not df2[feature].isnull().values.all():
        ax2.hist(df2[feature] , edgecolor = 'k', bins = bins)
    ax1.set_title(f'Histogram of {feature}')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Count')
    ax2.set_xlabel(feature)
    ax2.set_ylabel('Count')
    pls.show()

def pairwise_scatter_plot(df1,df2,feature,target, figsize=(12,6)):
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title(f'Scatterplot of {feature}'.format)
    ax1.set_xlabel(feature)
    ax1.set_ylabel(target)
    ax2.set_xlabel(feature)
    ax2.set_ylabel(target)
    if  not df1[feature].isnull().values.all():
        ax1.scatter(df1[feature], df1[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
    if  not df2[feature].isnull().values.all():
        ax2.scatter(df2[feature], df2[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
    pls.show()

def plot_pairwise_barplot(df1,df2,feature,**kwarg):
    """Plots pairwise feature barplot.
    """
    maxcount=kwarg.get('maxcount',20)
    temp = df1[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Count ': temp.values})

    temp = df2[feature].value_counts()
    df2 = pd.DataFrame({feature: temp.index,'Count ': temp.values})

    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=kwarg.get('figsize',(12,6)))
    sns.set_color_codes('pastel')

    if  not df1[feature].isnull().values.all():
        s = sns.barplot(ax=ax1, x = feature, y='Count ',order=df1[feature][:maxcount],data=df1[:maxcount])
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    if not df2[feature].isnull().values.all():
        s = sns.barplot(ax=ax2, x = feature, y='Count ',order=df2[feature][:maxcount],data=df2[:maxcount])
        s.set_xticklabels(s.get_xticklabels(),rotation=90)

    pls.tick_params(axis='both', which='major', labelsize=10)

    pls.show()

def plot_pairwise_stats(df1,df2,feature,target,**kwarg):
    """Plots pairwise feature statistics.
    """
    maxcount=kwarg.get('maxcount',20)

    fig, axs = pls.subplots(ncols=2,nrows=2,figsize=kwarg.get('figsize',(12,12)))

    if feature in df1.columns and not df1[feature].isnull().values.all():
        plot_stats(df1,feature,target,maxcount=maxcount,sortby='Count ',ax=axs[:,0],showfigure=False)

    if feature in df2.columns and not df2[feature].isnull().values.all():
        plot_stats(df2,feature,target,maxcount=maxcount,sortby='Count ',ax=axs[:,1])

    pls.show()

def pairwise_target_stat_per_day(df1,df2,target, **kwarg ):
    """Plots n top feature values counts per day

    Group dataframe by categorical feature value and calculates target statistics
    along datetime index . Plot calculated statics for both datasets side by side.

    Args:
        df1,df2   (str): datasets to compare
        target    (str): feature to average
    Keyword Args:
        figsize   (tuple): plot size
        linewidth (float): line width
        period      (str): sampling period
        method_name (str): sum,count,mean etc
.
    """
    figsize=kwarg.get('figsize',(20,4))
    linewidth=kwarg.get('linewidth',2.0)
    period=kwarg.get('period','1d')
    method_name=kwarg.get('method_name','mean')

    if df1.index.dtype==np.dtype('datetime64[ns]') and  df2.index.dtype==np.dtype('datetime64[ns]'):
        ax=getattr(df1[target].resample(period),method_name)().plot( grid=True,
                                                                    x_compat=True,
                                                                    figsize=figsize,
                                                                    linewidth=linewidth)
        getattr(df2[target].resample(period),method_name)().plot(ax=ax,
                                                                 grid=True,
                                                                 x_compat=True,
                                                                 figsize=figsize,
                                                                 linewidth=linewidth)
        ax.lines[0].set_linestyle(':')
        ax.lines[1].set_linestyle('--')
        pls.legend([f'First frame {target} {method_name}',f'Second frame {target} mean'])
        pls.show()


def display_side_by_side(dfs:list, captions:list)->None:
    """Displays two lists side by side."""
    output = ''
    attributes="style='display:inline'; font-size:110%'"
    combined = dict(zip(captions, dfs))
    styles   = [dict(selector='caption', props=[('text-align', 'left'),
        ('font-weight', 'bold'),('font-size', '120%'),('color', 'black')])]

    for caption, df in combined.items():
        output += df.style.set_table_attributes(attributes).set_caption(caption).set_table_styles(styles)._repr_html_()
        output += '\xa0\xa0\xa0'

    display(HTML(output))







