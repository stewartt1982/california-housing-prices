import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVR

class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
))


class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    Allows colomns to be selected by name from a pandas dataframe

    Parameters: list of str, names of the dataframe columns to select
    Default: []
    """
    def __init__(self, columns=[]):
        self.columns = columns
        
    """ Selects columns of a dataframe

    Parameters: X: pandas dataframe
    
    Returns: trans: pandas dataframe containing selected columns from X
    """
    def transform(self, X):
        trans = X[self.columns].copy()
        return trans

    """ Does nothing defined as it is needed
    Parameters: X : pandas dataframe
    y: default None

    Returns: self
    """
    def fit(self, X, y=None):
        return self


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    A transformer which is used to select columns in a pandas dataframe based on
    Type

    Parameters: datatype: pandas datatype
    """
    def __init__(self, datatype):
        self.datatype = datatype

    """
    does nothing definded as it is needed

    Parameters: X : pandas dataframe                                                                                                                    
    y: default None                                                                                            
     
    Returns: self 
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #assert instance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.datatype]).copy()

class AttributeDivider(BaseEstimator, TransformerMixin):
    def __init__(self, num_index, denom_index):
        self.num_index = num_index
        self.denom_index = denom_index

    def transform(self, X):
        ratio =  X[:,self.num_index]/X[:,self.denom_index]
        return np.c_[X,ratio]
    
    def fit(self,X, y=None):
        return self
    
#function to fet the data from the passed URL
def fetch_data(url="",path=""):
    if not os.path.isdir(path):
        #if the path to save the data does not exist
        #create the directory
        os.makedirs(path)
    #extract filename from the url
    urlpath = urllib.parse.urlparse(url)
    tgz_path = os.path.join(path,os.path.basename(urlpath.path))
    #get the file
    urllib.request.urlretrieve(url,tgz_path)
    #extract the file
    tgz = tarfile.open(tgz_path)
    tgz.extractall(path=path)
    tgz.close()

    
def load_data_csv(path="",file=""):
    csv_path = os.path.join(path,file)
    return pd.read_csv(csv_path)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

#fetch the data from the remote url
fetch_data(HOUSING_URL,"input")
#load the data into a dataframe
data_all = load_data_csv("input","housing.csv")

#We will now setup the pipeline to deal with the daa and
#prepare it for use

#What do we want to do?
#Deal with the categoical variable ocean_proximity by encodeing
#Impute missing values (if any)
#Scale data
#Aplly function, such as log, to data with long tails

data_all["ocean_proximity"]=data_all["ocean_proximity"].astype('category')
data_all["income_cat"] = np.ceil(data_all["median_income"] / 1.5)
data_all["income_cat"].where(data_all["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=13)
for train_index_strat2, test_index_strat2 in split.split(data_all, data_all["income_cat"]):
    train_set_strat2 = data_all.loc[train_index_strat2]
    test_set_strat2 = data_all.loc[test_index_strat2]
for set in (train_set_strat2, test_set_strat2):
    set.drop(["income_cat"], axis=1, inplace=True) 

#drop the target
train_set = train_set_strat2.drop("median_house_value",axis=1)
#save the target
train_target = train_set_strat2["median_house_value"].copy()
print train_set

#pipeline 1.
#Select features which are numerical and impute missing values by median
#columns for attributes
total_rooms = train_set.columns.get_loc("total_rooms")
households  = train_set.columns.get_loc("households")
total_bedrooms = train_set.columns.get_loc("total_bedrooms")
population = train_set.columns.get_loc("population")

NumPipeline = Pipeline([
    ('numerical',TypeSelector(np.number)),
    ('imputer',Imputer(strategy='median')),
    ('attrib_divider1',AttributeDivider(total_rooms,households)),
    ('attrib_divider2',AttributeDivider(total_bedrooms,total_rooms)),
    ('attrib_divider3',AttributeDivider(population,households)),
#    ('std_scaler',StandardScaler())
])



#pipeline 2.
#Select categorical features and convert them to onehotencodered numberical
CatPipeline = Pipeline([
    ('categoricals',TypeSelector('category')),
    ('stringindexer',StringIndexer()),
    ('cat_encoder', OneHotEncoder(sparse=False))
])



FullPipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", NumPipeline),
    ("cat_pipeline", CatPipeline)
]) 

full_output = FullPipeline.fit_transform(train_set)


#Now that we have the full provessed data we want to train a number of different models
#using either GridSearchCV or RandomSearchCV to optimise the hyper-parameters
#This is a simple, mall dataset so lets try both some quick simple estimators
#along with more complicated ones ie. slower
#Additionally lets try using DCA for dimensionality reduction for fun!
#Algo to be used:
#Linear Regression
#DecisionTree
#RandomForest
#SVR
#GradientBoostingTree
#create pipelines
#LinearRegression
pipe_LR = Pipeline([
    ('reg', LinearRegression())
])
param_grid = [
    {
        'reg': [LinearRegression()]
    },
    {
        'reg': [DecisionTreeRegressor()],
        "reg__min_samples_split": [2, 10, 20],
        "reg__max_depth": [None, 2, 5, 10],
        "reg__min_samples_leaf": [1, 5, 10],
        "reg__max_leaf_nodes": [None, 5, 10, 20],
    },
    {
        'reg': [RandomForestRegressor()],
        'reg__bootstrap': [True, False],
        'reg__n_estimators': [3, 10, 30],
        'reg__max_features': [2, 4, 6, 8]
    }
]

grid = GridSearchCV(pipe_LR, cv=5, n_jobs=1, param_grid=param_grid,verbose=2)
grid.fit(full_output,train_target)

#Lets start off by training a linear regression model
linreg = LinearRegression()
linreg.fit(full_output, train_target)
predictions = linreg.predict(full_output)
linmse = mean_squared_error(train_target, predictions) 
linrmse = np.sqrt(linmse)
print linrmse
dectreereg =  DecisionTreeRegressor()
dectreereg.fit(full_output, train_target)
predictions2 = dectreereg.predict(full_output)
dectreemse = mean_squared_error(train_target, predictions2) 
dectreermse = np.sqrt(dectreemse)
print dectreermse


#cross validation for the two models
linscores = cross_val_score(linreg,full_output,train_target,scoring="neg_mean_squared_error",cv=10)
linrmse_scores = np.sqrt(-linscores)

print linrmse_scores
print linrmse_scores.mean()
print linrmse_scores.std()


dectreescores = cross_val_score(dectreereg,full_output,train_target,scoring="neg_mean_squared_error",cv=10)
dectreermse_scores = np.sqrt(-dectreescores)

print dectreermse_scores
print dectreermse_scores.mean()
print dectreermse_scores.std()


#let us try to not overfit as much and try a RandomForest regression
rndforestreg =  RandomForestRegressor()
rndforestreg.fit(full_output, train_target)
predictions3 = rndforestreg.predict(full_output)
rndforestmse = mean_squared_error(train_target, predictions3)
rndforestrmse = np.sqrt(rndforestmse)
print rndforestrmse

rndforestscores = cross_val_score(rndforestreg,full_output,train_target,scoring="neg_mean_squared_error",cv=10)
rndforestrmse_scores = np.sqrt(-rndforestscores)

print rndforestrmse_scores
print rndforestrmse_scores.mean()
print rndforestrmse_scores.std()
