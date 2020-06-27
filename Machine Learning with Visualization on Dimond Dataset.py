#!/usr/bin/env python
# coding: utf-8

# ### Shubham Paliwal  -Business Analyst
# ### Machine Learning -  Project on Diamonds Dataset
# 

# ## Introduction

# This classic dataset contains the prices and other attributes of almost 54,000 diamonds. It's a great dataset for beginners learning to work with data analysis and visualization.

# ### Context:
# 
# price price in US dollars (\$326--\$18,823)
# 
# carat weight of the diamond (0.2--5.01)
# 
# cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# 
# color diamond colour, from J (worst) to D (best)
# 
# clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# 
# x length in mm (0--10.74)
# 
# y width in mm (0--58.9)
# 
# z depth in mm (0--31.8)
# 
# depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
# 
# table width of top of diamond relative to widest point (43--95)

# ### Importing Libraries:

# In[1]:


# Extended display of our data
import IPython, graphviz
from IPython.display import display
from sklearn import metrics

# Handle table-like data and matrices :
import pandas as pd
import numpy as np
import re
import math
import os
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

# Modelling Algorithms :
# Classification
rom sklearn.tree import export_graphviz
from sklearn.ensemble import forest
from IPython.display import Image
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ### Setting gateway path for data

# In[20]:


import os
os.chdir('C:\\Users\\Student\\Desktop\\ML End term')
os.getcwd()


# ### Read the dataset from the above path

# In[21]:


df_diamonds = pd.read_csv("diamonds.csv")


# ## Data Handling 

# ### EDA

# In[22]:


df_diamonds.head()


# In[23]:


## Here we're checking dimenstions of the dataset
df_diamonds.shape


# In[24]:


## Row max or columns max view 
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[25]:


# To check the data type which category each variables lies
df_diamonds.dtypes


# In[26]:


## Droping unnamed variable, which is actually not relavent for our further calculations.
df_diamonds.drop('Count', axis=1, inplace=True)


# In[27]:


## after droppping the serial number variable, we're checking the dataset head again.
df_diamonds.head()


# Chechking dataset summary (5 Number summary)

# In[38]:


df_diamonds.describe()


# In[28]:


# For plotting graphs 
import matplotlib.pyplot as plt        
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Null Value/Missing Value Treatment

# In[29]:


## Here we're checking that is there any null or NA or missing values in the dataset or else dataset is clean.
df_diamonds.isnull().sum()


# ## Data Distribution and spread

# ### Variables Understanding

# In[33]:


## Here we're  checking one by one each variable distribution. start with 'Cut'.
df_diamonds['cut'].value_counts()/df_diamonds['cut'].value_counts().sum()*100


# In[34]:


df_diamonds['cut'].value_counts().plot.barh()


# #### we can see that almost 40% of the data of cut is related to Ideal cut and rest 25 25 under premium & Very Good as per the distribution of the variable we can see that least important is Fair. but still we're not sure yet. let's go ahead...

# In[35]:


## Secondly, now we are taking color variable and checking its distribution.
df_diamonds['color'].value_counts()


# In[36]:


df_diamonds['color'].value_counts().plot.bar()


# #### Same as previously explained, G - Green color category got the highest weigtage in amount of numbers counting.

# In[40]:


## Now we are taking 'Clarity' variable and checking how it is distributed.
df_diamonds['clarity'].value_counts()


# In[41]:


df_diamonds['clarity'].value_counts().plot.barh()


# 
# #### Again, SI1 is on peak and got highest number of counts

# ## Basic Visualization:
# 
# ### Now here we're defining our Target Variable which is "Price", and try to observe its distribution and spread on the normal distributioncurve.
# 
# ### Target Variable - Price

# In[45]:


import seaborn as sns


# In[46]:


# Setting the plot size
fig, axis = plt.subplots(figsize=(7,7))
# We use kde=True to plot the gaussian kernal density estimate
sns.distplot(df_diamonds['price'], bins=50, kde=True)


# ### Here we're come up with the decent amount of information that data is highly right skewed and almost 40% of the prices of the given dataset variables is under the bin range of 0 to 5000.
# ### Here we need to work on it.

# In[56]:


## Continuous variable distribution, start with depth, then table, then carat.
df_diamonds['depth'].plot.box()


# In[60]:


# Setting the plot size
fig, axis = plt.subplots(figsize=(7,7))
# We use kde=True to plot the gaussian kernal density estimate
sns.distplot(df_diamonds['depth'], bins=20, kde=True)


# #### Depth - range of 55 to 65 cover large amount of data distribution.

# In[57]:


## Table - Distribution
df_diamonds['table'].plot.box()


# In[61]:


# Setting the plot size
fig, axis = plt.subplots(figsize=(7,7))
# We use kde=True to plot the gaussian kernal density estimate
sns.distplot(df_diamonds['table'], bins=30, kde=True)


# #### Table - more than 90% of table distribution is under the range of 50 to 70. fair enough, we'll move ahead.

# In[58]:


## Now Carat
df_diamonds['carat'].plot.box()


# ### It becomes hard to comment on single values of a variable which is continuous in nature and can having too many values, that too analysing it in uni-variate fashion and commenting on a trend or general behaviour becomes tough. So, here categorising the Price(s) comes to rescue and will help us understand the behaviour of Price in a much convinient way.

# In[68]:


# Convert target variable to categorical like low,medium,high and prime price and analyse: 

df_diamonds.loc[df_diamonds['price'] <= 5000,'Price_Cat'] = 0
df_diamonds.loc[(df_diamonds['price'] > 5000) & (df_diamonds.price <=10000),'Price_Cat'] = 1
df_diamonds.loc[(df_diamonds['price'] > 10000) & (df_diamonds.price <=15000),'Price_Cat'] = 2
df_diamonds.loc[df_diamonds['price'] > 15000,'Price_Cat'] = 3
df_diamonds.head(5)


# ### Breakpoints of 5000, 10000, 150000, and above 15000 is considered to be low, mid, high, and prime which is also clearly seen from the above histogram plot of Price.

# In[70]:


sns.countplot('Price_Cat',data=df_diamonds,color='blue')
plt.show()


# ### Here we can clearly see that there is nerly 37500 customers have their price in range 5000 to 10000. which is low range.

# In[1]:


## checking price effect on 'Cut' variable:
fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('education',data=df_diamonds,ax=axes[0],color='blue')
sns.countplot('education',hue='Price_Cat',data=df_diamonds,ax=axes[1])
plt.show()


# #### Fair enough, informative but still not found any conclusive statement.

# In[75]:


fig, axes = plt.subplots(1,2,figsize=(18,5))
sns.countplot('clarity',data=df_diamonds,ax=axes[0])
sns.countplot('clarity',hue='Price_Cat',data=df_diamonds,ax=axes[1])
plt.show()


# In[81]:


### Now here we are droping our new variable which was created to see the distribution more precisely "Price-Cat"
df_diamonds.drop('Price_Cat', axis=1, inplace=True)


# #### Again good enough to checking the distribution of clarity on the basis of price but still not so informative.m

# In[84]:


### Checking variable x, whcih is highly spread on the curve
# Setting the plot size
fig, axis = plt.subplots(figsize=(7,7))
# We use kde=True to plot the gaussian kernal density estimate
sns.distplot(df_diamonds['x'], bins=2, kde=True)


# In[85]:


# Setting the plot size
fig, axis = plt.subplots(figsize=(7,7))
# We use kde=True to plot the gaussian kernal density estimate
sns.distplot(df_diamonds['y'], bins=1, kde=True)


# ### Correlation Metrics 

# In[82]:


# Now, we're tring to see the correlation between the variables and try to check the multicollinearity of the variables.
Correlation = df_diamonds.corr(method='pearson')
Correlation


# In[83]:


# Generate a mask for the upper trinagle
# np.zeros_like_Return an array of zeros with the same shape
# and type as per given array
# In this case we pass the correlation matrix
# we create a varibale "mask" which is a 14 x 14 numpy array
mask = np.zeros_like(Correlation, dtype=np.bool)

# we create a tuple with triu_indices_from() passing the "mask" array
# k is used to offset diagonal
# with k=0, we offset all diagonal 
# if we put k=13, means we offset 14-13=1 diagonal

# triu_indices_from() return the indices for the upper-trinagle of err.
mask[np.triu_indices_from(mask, k=0)]=True

#Setting the plot size
fig, axis = plt.subplots(figsize=(11,11))
# cbar_kwn=("shrink":0.5) is shrinking the legend color bar
sns.heatmap(Correlation, mask=mask, cmap="YlGnBu", vmax=.4, center=0, square=True, linewidths=.1, cbar_kws={"shrink":0.5})


# In[86]:


links = Correlation.stack().reset_index()
links.columns = ['var1', 'var2','value']


# In[89]:


links_filtered=links.loc[ (links['value'] > 0.95) & (links['var1'] != links['var2']) ]
links_filtered


# In[155]:


# Creating a network graph to see the correlation
import networkx as nx
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
plt.subplots(figsize=(8,6))
nx.draw(G, with_labels=True, node_color='green', node_size=500, edge_color='black', linewidths=6, font_size=12)


# ### Now here comes to the very interesting part of the dataset, here we can see that these 4 variables are higly correlated with each other and with the excat correlation, strange but true, now we can also see the multicollinearity between the variables. we'll see the feature importance further.

# # FUNCTIONS TAB

# In[94]:



def set_plot_sizes(sml, med, big):
    plt.rc('font', size=sml)          # controls default text sizes
    plt.rc('axes', titlesize=sml)     # fontsize of the axes title
    plt.rc('axes', labelsize=med)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sml)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):	
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string or list of strings that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    >>>df2 = pd.DataFrame({'start_date' : pd.to_datetime(['3/11/2000','3/13/2000','3/15/2000']),
                            'end_date':pd.to_datetime(['3/17/2000','3/18/2000','4/1/2000'],infer_datetime_format=True)})
    >>>df2
        start_date	end_date    
    0	2000-03-11	2000-03-17
    1	2000-03-13	2000-03-18
    2	2000-03-15	2000-04-01
    >>>add_datepart(df2,['start_date','end_date'])
    >>>df2
    	start_Year	start_Month	start_Week	start_Day	start_Dayofweek	start_Dayofyear	start_Is_month_end	start_Is_month_start	start_Is_quarter_end	start_Is_quarter_start	start_Is_year_end	start_Is_year_start	start_Elapsed	end_Year	end_Month	end_Week	end_Day	end_Dayofweek	end_Dayofyear	end_Is_month_end	end_Is_month_start	end_Is_quarter_end	end_Is_quarter_start	end_Is_year_end	end_Is_year_start	end_Elapsed
    0	2000	    3	        10	        11	        5	            71	            False	            False	                False	                False	                False	            False	            952732800	    2000	    3	        11	        17	    4	            77	            False	            False	            False	            False	                False	        False	            953251200
    1	2000	    3	        11	        13	        0	            73	            False	            False	                False	                False               	False           	False           	952905600     	2000       	3	        11      	18  	5           	78          	False	            False           	False           	False               	False          	False           	953337600
    2	2000	    3	        11	        15	        2           	75          	False           	False               	False               	False               	False               False           	953078400      	2000    	4          	13      	1   	5           	92          	False           	True            	False           	True                	False          	False           	954547200
    """
    if isinstance(fldnames,str): 
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)
            
            
            
            
def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.
    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category
    """
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)



def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
  
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def fix_missing(df, col, name, na_dict):
    
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):

    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1
        
def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement.
    Parameters:
    -----------
    df: A pandas data frame, that you wish to sample from.
    n: The number of rows you wish to sample.
    Returns:
    --------
    return value: A random sample of n rows of df.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> get_sample(df, 2)
       col1 col2
    1     2    b
    2     3    a
    """
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))

def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))
    
def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
    
def parallel_trees(m, fn, n_jobs=8):
        return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))


# ## Pre Processing: 

# In[95]:


df_diamonds.price = np.log(df_diamonds.price)


# In[96]:


df, y, nas = proc_df(df_diamonds, 'price')


# In[97]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df,y)


# ### We're using random forest regression because our target variable is continous:

# ### Split data into 80:20 as train and validation sets.

# In[98]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 10800  # 80:20 as train and validation dataset distribution
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_diamonds, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# ## Random Forests
# ## Base model - Let's try our model again, this time with separate training and validation sets.
# 

# In[100]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[101]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[102]:


M = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
M.fit(X_train, y_train)
print_score(M)


# In[103]:


fi = rf_feat_importance(m, df); fi[:10]


# ## here now we come to the important information which is feature importance and we can see that Y is having highest weightage with .82 followed by carat which is 0.11. 
# ### Let's plot it.

# In[105]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[106]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[107]:


plot_fi(fi[:20]);


# In[108]:


### here we are keeping feature importance more than .005 and then check what our distrubtion says !!!
to_keep = fi[fi.imp>0.005].cols; len(to_keep)


# In[109]:


## Now we're keeping only these 5 variables into account for our next Random forest model.
df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[110]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[111]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi);


# ## Interesting observation in variable X and Clarity , previously where x is less important than clarity now it get more importance and in carat too previously it was soo far from y now it is quite close to Y.

# 
# ### Removing redundant features¶
# ### One thing that makes this harder to interpret is that there seem to be some variables with very similar meanings. Let's try to remove redundent features.

# In[112]:


import scipy
from scipy.cluster import hierarchy as hc


# In[113]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[148]:


S = RandomForestRegressor(n_jobs=-1, random_state=75, n_estimators=80, min_samples_leaf=2, 
                          max_features='auto')
get_ipython().run_line_magic('time', 'S.fit(X_train, y_train)')
print_score(S)


# In[121]:


P = RandomForestRegressor(n_jobs=-1, random_state=75, n_estimators=80, min_samples_leaf=4, max_features='auto')
get_ipython().run_line_magic('time', 'P.fit(X_train, y_train)')
print_score(P)


# In[147]:


Q = RandomForestRegressor(n_jobs=-1, random_state=75, n_estimators=75, min_samples_leaf=3, max_features=.75)
Q.fit(X_train, y_train)
print_score(Q)


# In[116]:


from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


# ### Now here we're using Extratrees classifier to check our model performance on more accurate scale.

# In[146]:


M = ExtraTreesRegressor(n_jobs=-1, random_state=75, n_estimators=75, min_samples_leaf=3, max_features=.75)
M.fit(X_train, y_train)
print_score(M)


# ### Here we can see that this is the best optimal point at which RMSE is .1911 and accuracy of validation dataset is 81.71%.

# In[149]:


M.score(X_valid, y_valid)


# In[154]:


results = pd.DataFrame({'Models by Shubham': ['RandomForestRegressor-S', 'RandomForestRegressor-P', 'RandomForestRegressor-Q','ExtraTreeRegressor'],
                        'Score': [S.score(X_valid, y_valid), P.score(X_valid, y_valid), Q.score(X_valid, y_valid), 
                                  M.score(X_valid, y_valid)]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# ### From the top 4 models, we can see that Regressor Q [no of trees =75, min_samples_leaf = 3 and max features used = .75] provides the highest accuracy of 73.22% in validation dataset.
# ### However, we have also considered ExtraTrees Regressor model for the same, where no of trees = 75, min samples leaf = 3 and max features used = .75, we can see the highest accuracy of 81.71% in validation dataset.

# # Thank You
