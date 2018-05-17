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
from sklearn.metrics import mean_squared_error

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

def print_gridsearch_cv(gridsearch,name="estimator"):
    print "Best Results: ",name,gridsearch.cv_results_['params'][gridsearch.best_index_]
    print "Best Score: ",name, np.sqrt(-gridsearch.best_score_)

def create_train_test(data=None,opt=2,split_size=0.2):
    #default opt=1 setting uses train_test_split
    #if opt=2 use tatified sampling
    data["ocean_proximity"]=data["ocean_proximity"].astype('category')
    if opt==1:
        train_set, test_set = train_test_split(data,test_size=split_size,random_state=13)
    elif opt==2:
        data["income_cat"] = np.ceil(data["median_income"] / 1.5)
        data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)
        split = StratifiedShuffleSplit(n_splits=1,test_size=split_size,random_state=13)
        for train_index_strat, test_index_strat in split.split(data, data["income_cat"]):
            train_set = data.loc[train_index_strat]
            test_set = data.loc[test_index_strat]
        #drop the income_cat variable used by StratifiedShuffleSplit
        for set in (train_set, test_set):
            set.drop(["income_cat"], axis=1, inplace=True) 

    return train_set, test_set

def get_target_set(train_set,test_set):
    #get set without the target
    train_set_notarget = train_set.drop("median_house_value",axis=1)
    test_set_notarget = test_set.drop("median_house_value",axis=1)
    #save the target
    train_target = train_set["median_house_value"].copy()
    test_target = test_set["median_house_value"].copy()
    
    return train_set_notarget,test_set_notarget,train_target,test_target


def create_pipeline():
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
        ('std_scaler',StandardScaler())
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

    return FullPipeline

def find_best_model_param(data=None,target=None):
    # pipe_LR = Pipeline([
    #     ('reg', LinearRegression())
    # ])
    # param_grid_LR = [
    #     {
    #         'reg': [LinearRegression()]
    #     }
    # ]
    # param_grid_DT = [
    #     {
    #         'reg': [DecisionTreeRegressor()],
    #         "reg__min_samples_split": [2, 10, 20],
    #         "reg__max_depth": [None, 2, 5, 10],
    #         "reg__min_samples_leaf": [1, 5, 10],
    #         "reg__max_leaf_nodes": [None, 5, 10, 20],
    #     }
    # ]
    # param_grid_RF = [
    #     {
    #         'reg': [RandomForestRegressor()],
    #         'reg__bootstrap': [True, False],
    #         'reg__n_estimators': [3, 10, 30],
    #         'reg__max_features': [2, 4, 6, 8]
    #     }
    # ]
    
    # grid_LR = GridSearchCV(pipe_LR, cv=5, n_jobs=1, param_grid=param_grid_LR,scoring='neg_mean_squared_error')
    # grid_LR.fit(data,target)
    # grid_DT = GridSearchCV(pipe_LR, cv=5, n_jobs=1, param_grid=param_grid_DT,scoring='neg_mean_squared_error')
    # grid_DT.fit(data,target)
    # grid_RF = GridSearchCV(pipe_LR, cv=5, n_jobs=1, param_grid=param_grid_RF,scoring='neg_mean_squared_error')
    # grid_RF.fit(data,target)

    # print_gridsearch_cv(grid_LR)
    # print_gridsearch_cv(grid_DT)
    # print_gridsearch_cv(grid_RF)

    best_model = FindBestModel()
    best_model.register_model_params(LinearRegression(),[{}],"Linear Regression Estimator")
    best_model.register_model_params(DecisionTreeRegressor(),[{"min_samples_split": [2, 10, 20],
                                                                "max_depth": [None, 2, 5, 10],"min_samples_leaf": [1, 5, 10],
                                                                "max_leaf_nodes": [None, 5, 10, 20]}],"Decision Tree Regression Estimator")
    best_model.register_model_params(RandomForestRegressor(),[{"bootstrap": [True, False],"n_estimators": [10, 50, 100, 500],
                                                                "max_features": [2, 4, 6, 8]}],"Random Forest Regression Estimator")
    best_estimator = best_model.evaluate_models(data,target)
    return best_estimator
    #return grid_RF.best_estimator_


class FindBestModel:
    def __init__(self):
        self.models = []
        self.params = []
        self.names = []
    def register_model_params(self,model,params,name):
        self.models.append(model)
        self.params.append(params)
        self.names.append(name)
    def evaluate_models(self,data,target):
        best_rmse = np.finfo(np.float128)
        best_model = None
        best_params = None
        model_results = []
        for model,param in zip(self.models,self.params):
            print model,param
            model_results.append(self.run_model(data,target,model,param))
        for param,score,best_estimator in model_results:
            print param,np.sqrt(-score)
            rmse=np.sqrt(-score)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = best_estimator
                best_params = param
        return best_estimator
    
    def run_model(self,data,target,model,params):
        results = GridSearchCV(model, params, cv=5, scoring="neg_mean_squared_error", n_jobs=1)
        results.fit(data,target)
        return [str(results.best_params_), results.best_score_, results.best_estimator_]

if __name__=='__main__':
    #Specify download path and location where data is stored
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "datasets/housing"
    HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
    
    #fetch the data from the remote url
    fetch_data(HOUSING_URL,"input")

    #load the data into a dataframe
    data_all = load_data_csv("input","housing.csv")

    #Create training set and test set

#We will now setup the pipeline to deal with the daa and
#prepare it for use

#What do we want to do?
#Deal with the categoical variable ocean_proximity by encodeing
#Impute missing values (if any)
#Scale data
#Apply function, such as log, to data with long tails

train_set_strat2,test_set_strat2 = create_train_test(data=data_all,opt=2,split_size=0.2)

#drop the target and save target
train_set,test_set,train_target,test_target = get_target_set(train_set_strat2, test_set_strat2)

FullPipeline = create_pipeline()
full_output = FullPipeline.fit_transform(train_set)
test_output = FullPipeline.transform(test_set)

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

final_model = find_best_model_param(full_output,train_target)
test_predictions = final_model.predict(test_output)
test_mse = mean_squared_error(test_predictions,test_target)
test_rmse = np.sqrt(test_mse)
print "Final RMSE: ",test_rmse
