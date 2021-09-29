import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as pls 

import seaborn as sns
from IPython.display import display, HTML

from .eda import plot_stats
from .eda import plot_cramer_v_corr,safe_convert,header
from .eda import plot_ntop_categorical_values_means
from .eda import plot_ntop_categorical_values_counts
from .eda import plot_ntop_categorical_values_sums
from .eda import plot_shaps,get_feature_info
from .eda import missing_values_table

#=====================#=====================#=====================
# dataset comparison  pairwise report
#=====================#=====================#=====================  

#def pairwise_report(df1,df2,target=None,ignore=[],nbrmax=20,full=True):
    # Compare two datasets:
    # df1      pandas dataframe     first dataset 
    # df2      pandas dataframe     second dataset
    # ignore  list                  features to ignore (optional)
    # nbrmax  int                   max number of top features (with max shap values) to print

#pairwise_report(df1,df2,target,ignore=[],nbrmax,full)

#creates html report
def pairwise_report(df1,df2,target=None,ignore=[],nbrmax=None,full=True,nbrmax20=20):
    
    if target!=None and target not in df.columns.to_list():
        print("{} not in dataframe - Ignore".format(target))
        target=None
        
    nbrmax=nbrmax if nbrmax!=None else max(df1.shape[1],df2.shape[1])
    
    #detect time columns
    df1 = df1.apply(lambda col: safe_convert(col) if col.dtypes == object else col, axis=0)
    df2 = df2.apply(lambda col: safe_convert(col) if col.dtypes == object else col, axis=0)

    #print shap values for each frame predicting targed
    header('Shap values' )
     
    if target==None:#compare datasets
        sorted_features=plot_shap_pairwise(df1,df2,ignore,nbrmax)
    else:           #compare acconding to target
        sorted_features1=plot_shap(df1,target)
        sorted_features2=plot_shap(df2,target)
        sorted_features=sorted_features1 if len(sorted_features1)>len(sorted_features2) else sorted_features2

    sorted_features=sorted_features[:nbrmax]
 
    # if dataframe has timedate index - plot time series
    if target !=None and  df1.index.dtype==np.dtype('datetime64[ns]') and df2.index.dtype==np.dtype('datetime64[ns]'):
        #header again
        header('Time series') 
        #print frames records counts per day side by side
        pairwise_feature_stat_per_day(df1,df2,target,method_name='count')
        
    #print each feature stat
    header('Features info' )
    print_features_pairwise(df1,df2,target,sorted_features)
       
    if full:
        header('Missed values')
        #plot missed vales
        plot_na_pairwise(df1,df2)
        print('\n \n ')
        #print columns with % of missing values
        print_na_pairwise(df1,df2)

        #for numeric variables only
        header('Pearson correlations' )     
        plot_correlations_pairwise(df1,df2,nbrmax20) 

        #correlations of categorical variables
        header('Cramers V staticstics' )    
        #third parameter max features to display
        plot_cramer_v_corr_pairwise(df1,df2,nbrmax20)    

def print_features_pairwise(df1,df2,target=None,sorted_features=[]):
        
    features = sorted_features if len(sorted_features)>0 else list(set(df1.columns.to_list()) & set(df2.columns.to_list()))

    for feature in features:
        
        if feature==target:
            continue
            
        print('\n ')
        
        info1 = pd.DataFrame(
                index=['Type :' ,'Distinct count :', 'Missed %:'],
                columns=[' '])
        info2 = pd.DataFrame(
                index=['Type :' ,'Distinct count :', 'Missed %:'],
                columns=[' '])
    
        header(feature,sz='h3')
        
        info1[' '] = get_feature_info(df1,feature)
        
        if info1.iloc[0,0]=='Numeric':
            info1.loc['Mean :',' ']=df1[feature].mean()
        
        info2[' '] = get_feature_info(df2,feature)
        if info2.iloc[0,0]=='Numeric':
            info2.loc['Mean :',' ']=df2[feature].mean()

        display_side_by_side([info1.head(),info2.head()],['Frame 1','Frame 2'])
        print('\n ') 

        noempty1=info1.loc['Distinct count :'][' ']>0
        noempty2=info2.loc['Distinct count :'][' ']>0
        
        if info1.iloc[0,0]=='Categorical' or info1.iloc[0,0]=='Boolean':
            if info1.iloc[1,0]>df1.shape[0]/2.0 :
                print("Too many values to plot ")
            elif df1[feature].isnull().values.all() and df2[feature].isnull().values.all():
                print("All values are null")
            elif info1.iloc[1,0]<2:
                print("Zero variance")
            else:
                plot_pairwise_stats(df1,df2,feature,target,30)
                #count of records with feature=value per day
                if target != None and  df1.index.dtype==np.dtype('datetime64[ns]') and df2.index.dtype==np.dtype('datetime64[ns]'):
                    display(HTML("<h3 align=\"center\">Top {} count per day</h3>".format(feature)))
                    if noempty1: plot_ntop_categorical_values_counts(df1,feature,target,4)
                    if noempty2: plot_ntop_categorical_values_counts(df2,feature,target,4)
                
                    #mean of target for records with feature=value per day
                    display(HTML("<h3 align=\"center\">{} mean per day </h3>".format(target)))
                    if noempty1: plot_ntop_categorical_values_means(df1,feature,target,4)
                    if noempty2: plot_ntop_categorical_values_means(df2,feature,target,4)
                
        elif info1.iloc[0,0]=='Numeric':
            #pairwise_feature_stat_per_day(df1,df2,feature,method_name='count')
            #pairwise_feature_stat_per_day(df1,df2,feature,method_name='mean')
            if info1.iloc[1,0]<=25:
                plot_pairwise_stats(df1,df2,feature,target,30)
            else:
                #pairwise_density_plot(df1,df2,feature)  
                pairwise_density_plot_ex(df1,df2,feature)
                
                fig,ax = pls.subplots(1, 2,figsize=(16, 5))
                if noempty1: sns.boxplot(df1[feature],color='blue',orient='h',ax=ax[0])                
                if noempty2: sns.boxplot(df2[feature],color='blue',orient='h',ax=ax[1])
                pls.show()
                
                if  target !=None:
                    pairwise_scatter_plot(df1,df2,feature,target)
                    fig,ax = pls.subplots(1, 2,figsize=(12, 6))
                    if noempty1: sns.violinplot(x=target, y=feature, data=df1, ax=ax[0], inner='quartile')
                    if noempty2: sns.violinplot(x=target, y=feature, data=df2, ax=ax[1], inner='quartile')
                    pls.show()                        
                    
        else:
            print("Time column - skip  ")
    

#=====================#=====================#=====================#=====================
# shap 
#=====================#=====================#=====================#=====================
       
def plot_shap_pairwise(df1, df2,ignore=[],nbrmax=20):
    
    target='fake_target_547454889'
    x=df1.copy()
    x[target]=0
    xx=df2.copy()
    xx[target]=1
    x=pd.concat([x,xx],axis=0)

    return plot_shaps(x,target,ignore,nbrmax)

def plot_cramer_v_corr_pairwise(df1,df2,max_features=20):
    fig, ax = pls.subplots(1,2,figsize=(40,20))
    plot_cramer_v_corr(df1,10,ax[0])
    plot_cramer_v_corr(df2,10,ax[1]) 
    pls.show()
    
def plot_correlations_pairwise(df1,df2,maxnbr=20):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    l1=df1.select_dtypes(include=numerics).columns.to_list()
    l2=df2.select_dtypes(include=numerics).columns.to_list()
    if len(l1)>=2 or len(l2)>=2:
        fig, ax = pls.subplots(1,2,figsize=(40,20))
        sns.heatmap(df1[l1[:maxnbr]].corr(), cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6,ax=ax[0])
        ax[0].set_title('Correlation Heatmap')
        sns.heatmap(df2[l2[:maxnbr]].corr(), cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6,ax=ax[1])
        ax[1].set_title('Correlation Heatmap')   
        pls.show()
    else:
        print('Less then 2 numeric columns - skip ')
    
def print_na_pairwise(df1,df2,max_row=60):
    display_side_by_side([missing_values_table(df1).head(max_row),missing_values_table(df2).head(max_row)],['Frame 1 ','Frame 2'])
    
def plot_na_pairwise(df1,df2):
    
    pls.style.use('seaborn-talk')

    fig = pls.figure(figsize=(18,6))
    miss_1 = pd.DataFrame((df1.isnull().sum())*100/df1.shape[0]).reset_index()
    miss_2 = pd.DataFrame((df2.isnull().sum())*100/df2.shape[0]).reset_index()
    miss_1["type"] = "first"
    miss_2["type"]  =  "second"
    missing = pd.concat([miss_1,miss_2],axis=0)
    ax = sns.pointplot("index",0,data=missing,hue="type")
    pls.xticks(rotation =90,fontsize =7)
    pls.title("Missed data")
    pls.ylabel(" %")
    pls.xlabel("Features")
    pls.show()
    
def pairwise_density_plot(df1,df2,feature, figsize=(12,6)):
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    if  not df1[feature].isnull().values.all():sns.distplot(df1[feature], hist = False,ax=ax1)
    if  not df2[feature].isnull().values.all():sns.distplot(df2[feature], hist = False,ax=ax2)
    pls.title('Density Plot and Histogram of {}'.format(feature))
    pls.show()
    
def pairwise_density_plot_ex(df1,df2,feature, figsize=(12,6)):
    fig,(ax1, ax2)  = pls.subplots(1, 2,figsize=(12, 6))
    if  not df1[feature].isnull().values.all():
        sns.distplot(df1[feature],kde=True,ax=ax1) 
    ax1.axvline(df1[feature].mean(),color = "k",linestyle="dashed",label="MEAN")
    ax1.legend(loc="upper right")
    ax1.set_title('{} Skewness = {:.4f}'.format('Frame 1',df1[feature].skew()))

    if  not df2[feature].isnull().values.all():
        sns.distplot(df2[feature],kde=True,ax=ax2) 
    ax2.axvline(df2[feature].mean(),color = "k",linestyle="dashed",label="MEAN")
    ax2.legend(loc="upper right")
    ax2.set_title('{} Skewness = {:.4f}'.format('Frame 2',df2[feature].skew()))
    pls.show()
    
def pairwise_hist(df1,df2,feature,bins=30, figsize=(12,6)):
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    if  not df1[feature].isnull().values.all(): ax1.hist(df1[feature] , edgecolor = 'k', bins = bins)
    if  not df2[feature].isnull().values.all(): ax2.hist(df2[feature] , edgecolor = 'k', bins = bins)
    ax1.set_title('Histogram of {}'.format(feature)); 
    ax1.set_xlabel(feature); 
    ax1.set_ylabel('Count');
    ax2.set_xlabel(feature); 
    ax2.set_ylabel('Count');
    pls.show()
    
def pairwise_scatter_plot(df1,df2,feature,target, figsize=(12,6)):
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title('Histogram of {}'.format(feature)); 
    ax1.set_xlabel(feature); 
    ax1.set_ylabel(target);
    ax2.set_xlabel(feature); 
    ax2.set_ylabel(target);
    if  not df1[feature].isnull().values.all():ax1.scatter(df1[feature], df1[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
    if  not df2[feature].isnull().values.all():ax2.scatter(df2[feature], df2[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
    pls.show()     

#=====================#=====================#=====================#=====================
# categorical 
#=====================#=====================#=====================#=====================

def __plot_pairwise_stats(df_1,df_2,feature,end=30):

    temp = df_1[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Count ': temp.values})

    temp = df_2[feature].value_counts()
    df2 = pd.DataFrame({feature: temp.index,'Count ': temp.values})

    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=(12,6))    
    sns.set_color_codes("pastel")
    
    if  not df_1[feature].isnull().values.all():
        s = sns.barplot(ax=ax1, x = feature, y="Count ",order=df1[feature][:end],data=df1[:end])
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    if not df_2[feature].isnull().values.all():        
        s = sns.barplot(ax=ax2, x = feature, y="Count ",order=df2[feature][:end],data=df2[:end])
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
        
    pls.tick_params(axis='both', which='major', labelsize=10)
    
    pls.show();

def plot_pairwise_stats(df1,df2,feature,target,end=30):
    
    if target != None:
        
        fig, axs = pls.subplots(2,2,figsize=(16,14))

        if feature in df1.columns and not df1[feature].isnull().values.all():
            plot_stats(df1,feature,target,end,'Count ',ax1=axs[0,0],ax2=axs[1,0])

        if feature in df2.columns and not df2[feature].isnull().values.all():    
            plot_stats(df2,feature,target,end,'Count ',ax1=axs[0,1],ax2=axs[1,1])

        pls.show();
    else:
        __plot_pairwise_stats(df1,df2,feature,end=30)
        

#=====================#=====================#=====================
# time series plots
#=====================#=====================#=====================
def pairwise_feature_stat_per_day(df1,df2,target,figsize=(20,4),linewidth=2.0,method_name='mean',period="1d"):
    
    if df1.index.dtype==np.dtype('datetime64[ns]') and  df2.index.dtype==np.dtype('datetime64[ns]'):
        ax=getattr(df1[target].resample(period),method_name)().plot( grid=True,x_compat=True,figsize=figsize,linewidth=linewidth)
        getattr(df2[target].resample(period),method_name)().plot(ax=ax, grid=True, x_compat=True, figsize=figsize, linewidth=linewidth)
        ax.lines[0].set_linestyle(":")
        ax.lines[1].set_linestyle("--")
        pls.legend(['First frame {} {}'.format(target,method_name),'Second frame {} mean'.format(target)])
        pls.show()
        
        
def display_side_by_side(dfs:list, captions:list):
    
    output = ""
    attributes="style='display:inline'; font-size:110%'"
    combined = dict(zip(captions, dfs))
    styles   = [dict(selector="caption", props=[("text-align", "left"),
        ("font-weight", "bold"),("font-size", "120%"),("color", 'black')])] 
    
    for caption, df in combined.items():
        output += df.style.set_table_attributes(attributes).set_caption(caption).set_table_styles(styles)._repr_html_()
        output += "\xa0\xa0\xa0"
        
    display(HTML(output))

#for compatibility
print_report = pairwise_report  


       
        
        


