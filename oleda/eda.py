import numpy as np
import matplotlib.pyplot as pls 
import pandas as pd
import warnings

from IPython.display import display, HTML
import seaborn as sns

from .eda_core  import *
from .eda_anova import anova,two_way_anova,turkeyHSD

warnings.filterwarnings("ignore")
    
#=====================#=====================#=====================
# single dataset  eda
#=====================#=====================#=====================  

#single dataset report
def report(df,target=None,ignore=[],nbrmax=20):
    do_eda(df,target,ignore,nbrmax)  

#shap values
def  plot_shaps(df, target, ignore=[], nbrmax=20):
    #
    # plot shaps values
    # target should be binary or float
    # returns features list sorted by importance
    #    
    return plot_shap( df, target, ignore, nbrmax)
    
#=====================#=====================#=====================#=====================
# numerical continues
#=====================#=====================#=====================#=====================

def plot_cuts(df,feature,target,bins=None, figsize=(12,6)):
    
    if bins==None:
        bins=np.arange(df[feature].min(),df[feature].max(),(df[feature].max()-df[feature].min())/10.)
    
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title('Histogram of {}'.format(feature)); 
    ax1.set_xlabel(feature)
    ax1.set_ylabel('count')
    ax2.set_xlabel(feature)
    ax2.set_ylabel(target)
    df.groupby(pd.cut(df[feature], bins=bins))[target].count().plot(kind='bar',ax=ax1,grid=True)
    df.groupby(pd.cut(df[feature], bins=bins))[target].mean().plot(kind='bar',ax=ax2,grid=True)  
    pls.show()  
    
def plot_qcuts(df,feature,target,q=None, figsize=(8,4)):
    
    if q==None:
        q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1]
    
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title('Histogram of {}'.format(feature)); 
    ax1.set_xlabel(feature) 
    ax1.set_ylabel('count')
    ax2.set_xlabel(feature) 
    ax2.set_ylabel(target)
    df.groupby(pd.qcut(df[feature], q=q,duplicates='drop'))[target].count().plot(kind='bar',ax=ax1,grid=True)
    df.groupby(pd.qcut(df[feature], q=q,duplicates='drop'))[target].mean( ).plot(kind='bar',ax=ax2,grid=True)

    pls.show()    
                      
#=====================#=====================#=====================#=====================
# categorical 
#=====================#=====================#=====================#=====================

def plot_stats(df,feature,target,max_nbr=20,sort='Count ',ax1=None,ax2=None):
    
    end=max_nbr
    createfig=(ax1==None or ax2==None)
    cat_count = df[feature].value_counts().reset_index()
    cat_count.columns = [feature,'Count ']
    cat_count.sort_values(by=sort, ascending=False, inplace=True)

    cat_perc = df[[feature, target]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_count,on=feature)
    cat_perc.sort_values(by=sort, ascending=False, inplace=True)
    
    size=(12,6) if len(cat_count[:max_nbr]) <=40 else (12,14)
    if createfig:
        fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=size)
        
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Count ",order=cat_count[feature][:max_nbr],data=cat_count[:max_nbr])
    s.set_xticklabels(s.get_xticklabels(),rotation=90)   
    
    s = sns.barplot(ax=ax2, x = feature, y=target, order=cat_perc[feature][:max_nbr], data=cat_perc[:max_nbr])  
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
        
    pls.ylabel(target, fontsize=10)
    pls.tick_params(axis='both', which='major', labelsize=10)

    if createfig:
        pls.show()
        
def plot_melt(df,feature,target1,target2,end=20):
    
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

def plot_na(df):
    
    #pls.style.use('seaborn-talk')

    fig = pls.figure(figsize=(18,6))
    miss = pd.DataFrame((df.isnull().sum())*100/df.shape[0]).reset_index()

    ax = sns.pointplot("index",0,data=miss)
    pls.xticks(rotation =90,fontsize =7)
    pls.title("Missed data")
    pls.ylabel(" %")
    pls.xlabel("Features")
    
def print_na(df,maxnbr=None):
    
    max_row=max_row=df.shape[0] if maxnbr==None else maxnbr
    
    mdf=missing_values_table(df)
    if mdf.shape[0]:
        print(missing_values_table(df).head(max_row))
    else:
        print('No missed values in dataframe ')
    
#=====================#=====================#=====================#=====================
# correlations
#=====================#=====================#=====================#=====================
    
def corr(df,maxnbr=20,figsize=None):
    #
    # plots correlation heatmap for all numerical features
    #
    corr=df.corr()
    cx=corr.abs().unstack().sort_values(ascending=False).reset_index().dropna() 
    lst=list(cx[cx.level_0!=cx.level_1]['level_0'].drop_duplicates(keep='first')[:maxnbr].values)
    #show most correlated values
    if figsize==None:
        size=min(len(lst),20)
        figsize=(size,size)
    fig, ax = pls.subplots(1,1,figsize=figsize)
    sns.heatmap( corr[corr.index.isin(lst)][lst], cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    pls.title('Correlation Heatmap')
    pls.show() 
    return cx

#plots correlation heatmap for features from the list 
#show all features correlation without filtering
def corr_x(df,features,figsize=(20,20)):
    if len(features)>=2:
        fig, ax = pls.subplots(1,1,figsize=figsize)            
        sns.heatmap(df[features].corr(), cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
        pls.title('Correlation Heatmap')
        pls.show()
        
#plots correlation heatmap between features from two lists        
def corr_2x(df,features_x,features_y,figsize=(20,20)):
    c=features_x.copy()
    c.extend(features_y)
    fig, ax = pls.subplots(1,1,figsize=figsize)
    sns.heatmap(df[c].corr()[features_y].T[features_x], cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    pls.title('Correlation Heatmap')
    pls.show()    
       
#=====================#=====================#=====================#=====================
# report
#=====================#=====================#=====================#=====================

def print_features(df,target=None,sorted_features=[]):

    #explore features selected by shap (sorted_features)
    features = sorted_features if len(sorted_features)>0 else df.columns.to_list()

    _,tg_cardinality,_ = get_feature_info(df,target)
   
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    numeric=list(set(numeric)&set(sorted_features))

    fanova=[]
            
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
            
        info = pd.DataFrame(
        index=['Type :' ,'Distinct count :', 'Missed %:'],
        columns=[' '])       
        info[' ']=[feature_type,cardinality,missed ]
        print(info.head())
        print('\n ') 
        
        #-----------------Categorical
        
        if feature_type=='Categorical' or feature_type=='Boolean':
            if feature_type=='Categorical':
                    df[feature]=df[feature].fillna('Missed value')
              
            if cardinality > df.shape[0]/2.0 :
                print("Too many values to plot ")
            elif df[feature].isnull().values.all() :
                print("All values are null")
            elif cardinality<2:
                print("Zero variance")
                                                                   
            else:
                if target != None :
                    plot_stats(df,feature,target,30)
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
                    plot_ntop_categorical_values_counts(df,feature,target,4)
                
                    #mean of target for records with feature=value per day
                    display(HTML("<h3 align=\"center\">{} mean per day </h3>".format(target)))
                    plot_ntop_categorical_values_means(df,feature,target,4)
                 
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
                top=df[feature].value_counts().iloc[:4].index
                sub=df[df[feature].isin(top)]

                for t in numeric:
                    try:
                        #check variance 
                        if anova(sub,feature,t,False):
                            depended.append(t)
                    except Exception:
                        a=0

                if len(depended)>0:
                    print('Anova passed for ',depended)
                    fanova.append(feature)  
                    if len(depended)>1:
                        depended=get_top_correlated(sub[depended])
                        if len(depended)>1:
                            sns.pairplot(sub,vars=depended,hue=feature,corner=True)
                            pls.show()

            if feature_type=='Categorical':
                    df[feature]=df[feature].replace('Missed value',np.nan)      
        #---------------- Numeric    
        
        elif feature_type=='Numeric':
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
                    plot_stats(df,feature,target,40)
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
                    #.sample(min(df.shape[0],100000))
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
                        ax[0].scatter(df[feature], df[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
                        ax[0].set_xlabel(feature)
                        ax[0].set_ylabel(target)
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
                        
                       
        else:
            info = pd.DataFrame(
            index=['dType :' ,'Min :', 'Max :'],
            columns=[' '])       
            info[' ']=[df[feature].dtype,df[feature].min(),df[feature].max()]
            print(info.head())
            print('\n ')
            
            print("Time column skip plotting ")
            
                                            
def do_eda(df,target,ignore=[],nbrmax=20,figsize=(20,4),linewidth=2):
    
    if target!=None and target not in df.columns.to_list():
        print("{} not in dataframe - Ignore".format(target))
        target=None
    #detect time columns
    df = df.apply(lambda col: safe_convert(col) if col.dtypes == object else col, axis=0)
    
    #print nans statistics
    header('Missed values' )
    print_na(df)
                      
    #print shap values for each frame predicting targed
    feature_type,cardinality,missed = get_feature_info(df,target)
    
    if feature_type=='Numeric' :
        header('Shap values')
        sorted_features=plot_shap(df,target,ignore=ignore,nbrmax=nbrmax)#[:nbrmax]
    else:
        sorted_features=[f for f in df.columns.tolist() if f not in ignore]
        #list(set(df.columns.tolist())-set(ignore))


    # if dataframe has timedate index - plot time series
    if target !=None and  isTime(df.index.dtype) and df.index.nunique()> 2:
        header('Time series' )
        ax=df[target].resample('1d').mean().plot( grid=True,x_compat=True,figsize=figsize,linewidth=linewidth,label=target)
        pls.title(' {} mean per day '.format(target))
        pls.legend()
        pls.show() 
    
    
    header('Features' )
    print_features(df,target,sorted_features)
     
    #for numeric variables only
    header('Pearson correlations' )     
    cx=corr(df,nbrmax) 

    #correlations of categorical variables
    header('Cramers V staticstics' )    
    #third parameter max features to display
    plot_cramer_v_corr(df,max_features=nbrmax,ignore=ignore) 

    #plot 10 most correlated features
    header('Top correlated features' )  
    f=list(cx[(cx[0]>0.099)&(cx.level_0!=cx.level_1)]['level_0'].drop_duplicates(keep='first')[:6].values)
    g=sns.pairplot(df[f])  
    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation = 90)
        ax.set_ylabel(ax.get_ylabel(), rotation = 0)
        ax.yaxis.get_label().set_horizontalalignment('right')
    
    return sorted_features

def get_top_correlated(df,th=0.099,maxcount=10):
    
    corr=df.corr()
    if corr.shape[0]>0:
        cx=corr.unstack().dropna().abs().sort_values(ascending=False).reset_index()
        return list(cx[(cx[0]>th)&(cx.level_0!=cx.level_1)]['level_0'].drop_duplicates(keep='first')[:maxcount].values)   
    else:
        return []
    

def one_to_many(df,fiterout=True):
    
    features =  df.columns.to_list()
    
    for f11 in  range(len(features)):
        for f22 in range(f11+1,len(features)):
            c2=df.groupby(features[f11])[features[f22]].nunique().max()
            c1=df.groupby(features[f22])[features[f11]].nunique().max()
            if not fiterout or c1<2 or c2<2:
                print(features[f11],' - ',features[f22],'\n\t relation ',c1,' : ',c2)
            
def interactions2x(ddf,feature=[],target=[],maxnbr=4,bins=True):
    
    df=ddf.copy()
    
    with pd.option_context('mode.use_inf_as_na', True):    
        if df.isnull().values.any():
            print('Warning: dataframe contains nan or inf , please fix or drop them to obtain better results. \n')
            df=df.fillna(0)
    
    features =  df.columns.to_list() if len(feature)==0 else feature
   
    candidates=[]
    numeric=target.copy()
    
    for f in features:                                                                                                                                                       
        feature_type,cardinality,missed = get_feature_info(df,f)
        
        if feature_type=='Categorical' or feature_type=='Boolean' and\
            cardinality < df.shape[0]/2.0 and not df[f].isnull().values.all()\
            and cardinality>=2:
            candidates.append(f)
                                                                   
        elif feature_type=='Numeric':
           
            if len(target)==0 and cardinality> 1:
                numeric.append(f)
                
            if cardinality < df.shape[0]/2.0 and cardinality>=2 and cardinality < 40:
                candidates.append(f)
            elif bins:
                #q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                #df['cuts__'+f]=pd.qcut(df[f], q=q,duplicates='drop')
               
                df['cuts__'+f]=pd.cut(df[f], bins=[0,1,2,3,5,10,15,20,60,100,1000])
                df['cuts__'+f]=df['cuts__'+f].astype(str)
                
                candidates.append('cuts__'+f)
    fanova={}
   
    if  len(candidates)<1:
        return fanova      

    i=0
    for f1 in candidates:
            i=i+1
            top=df[f1].value_counts().iloc[:maxnbr].index
            sub=df[df[f1].isin(top)].copy()
            sub['count']=1
            if False:
                for f2 in candidates[i:]:
                    if f1==f2:
                        continue
                        
                    header(f1+' - '+ f2,sz='h3')
                    print('relation ',df.groupby(f1)[f2].nunique().max(),' : ',df.groupby(f2)[f1].nunique().max())

                    top2=sub[f2].value_counts().iloc[:maxnbr].index
                    sub2=sub[sub[f2].isin(top2)].copy()
                    ds=sub2.groupby([f1,f2])['count'].count().reset_index()
                    
                    if ds['count'].min()/ds['count'].max()<0.8 :            
                        fig, ax = pls.subplots(figsize=(8, 6)) 
                        sns.barplot(x=f1, y='count', hue=f2, data=ds,ax=ax)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                        pls.show()
            #to speed up - select 4 most popopular values
            
            if len(numeric)<1 :
                continue

            depended=[]
            for t in numeric:
                if ('cuts__'+t) ==f1 or t ==f1:
                    continue
                    
                #print(f1,' - ',t)
                try:
                    #check variance 
                    if anova(sub,f1,t,False):
                        header(f1+' - '+ t,sz='h3')
                        depended.append(t)
                        if maxnbr>6:
                            fig, ax = pls.subplots(figsize=(14, 8))
                        else:
                            fig, ax = pls.subplots(figsize=(8, 6))                            
                        sns.barplot(x=f1, y=t, data=sub,ax=ax)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                        pls.show()
                        turkey=turkeyHSD(sub,f1,t)
                        if turkey.shape[0]>0:
                            print('turkeyHSD')
                            print(turkey)
                            print('\n')
                except Exception as e:
                    a=0
                    #print(e)
          
            if len(depended)>0:
                print(f1, ' - ',depended,'\n\n\n') 
                fanova[f1]= depended
                depended=get_top_correlated(sub[depended])

                if len(depended)>1:               
                    sns.pairplot(sub,vars=depended,hue=f1,corner=True)
                    pls.show()
    return fanova


def interactions3x(ddf,feature=[],target=[],verbose=False,maxnbr=6):
    
    df=ddf.copy()
    
    with pd.option_context('mode.use_inf_as_na', True):    
        if df.isnull().values.any():
            print('Warning: dataframe contains nan or inf , please fix or drop them to obtain better results. \n')
            df=df.fillna(0)
    
    features = df.columns.to_list() if len(feature)==0 else feature
   
    candidates=[]
    numeric=target.copy()
    
    for f in features:                                                                                                                                                       
        feature_type,cardinality,missed = get_feature_info(df,f)
        
        if (feature_type=='Categorical' or feature_type=='Boolean') and\
            cardinality < df.shape[0]/2.0 and not df[f].isnull().values.all()\
            and cardinality>=2:
            candidates.append(f)
                                                                   
        elif feature_type=='Numeric':
            if len(target)==0 and cardinality> 1:
                numeric.append(f)
                
            if cardinality < df.shape[0]/2.0 and cardinality>=2 and cardinality < 40:
                candidates.append(f)
            else:
                q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1]
                df['__'+f]=pd.qcut(df[f], q=q,duplicates='drop')
                df['__'+f]=df['__'+f].astype(str)
                
                candidates.append('__'+f)
                
    for ff1 in range(len(candidates)-1):
        f1=candidates[ff1]
        top1=df[f1].value_counts().iloc[:maxnbr].index
        for ff2 in range(ff1+1,len(candidates)):
            sub1=df[df[f1].isin(top1)]
            f2=candidates[ff2]
            header(f1+' - '+ f2,sz='h3')
            
            #one-to-one
            if df.groupby(f1)[f2].apply(lambda x: x.nunique() <2 ).all():
                print ('one-to-one skip')
                continue
                
            print('relation ',df.groupby(f1)[f2].nunique().max(),' : ',df.groupby(f2)[f1].nunique().max())
            sub1[f1+f2]=sub1[f1].astype(str)+sub1[f2].astype(str)
            
            top=sub1[[f1,f2,f1+f2]].dropna()[f2].value_counts()[:maxnbr].index
            sub=sub1[sub1[f2].isin(top)]
            
            depended=[] 
            for t in numeric:
                print('\n\n',t)
                if ('__'+t) in [f1, f2] or t in [f1,f2]:
                    continue

                try:
                                        
                    res=two_way_anova(sub[[f1,f2,f1+f2,t]].dropna(),f1,f2,t)
                    if verbose:
                        print(res)

                    if(res.iloc[2]['PR(>F)']<=0.05):
                        print('Anova passed')
                        #print(f1,f2,t)
                        depended.append(t)
                        if maxnbr>6:
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
                            if maxnbr>6:
                                fig, ax = pls.subplots(figsize=(14, 8))
                            else:
                                fig, ax = pls.subplots(figsize=(8, 6)) 
                            bar=sns.barplot(x=f1, y=t, hue=f2, data=sub,ax=ax)
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                            pls.show()                            
                        print('Anova faled to reject => no difference ')

                except Exception as e:
                    print('nan',e)
                    a=0
            if len(depended)>0:        
                print(depended)     
            
            if len(depended)>0:         
                if len(depended)>1: 
                    top2=sub[[f1,f2,f1+f2]].dropna()[f1+f2].value_counts().index[:maxnbr]
                    sns.pairplot(sub[sub[f1+f2].isin(top2)],vars=depended,hue=f1+f2,corner=True)
                    pls.show()
                for t in depended:
                    print(f1+'*'+f2+'='+t)
                    g = sns.catplot(x=t, y=f1, row=f2, kind="box", orient="h", height=1.5, aspect=4,data=sub)
                    pls.show()