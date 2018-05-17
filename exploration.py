import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor

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
data=data_all.copy()
data["ocean_proximity"]=data["ocean_proximity"].astype('category')

#lets do some basic checks on the data
print data.head()
print data.info()
print data.describe()
#ocean_proximity is not numerical. Later we will use a pipeline to deal
#with this categorical variable
print data['ocean_proximity'].value_counts()

data.hist(bins=50,figsize=(20,15))
plt.savefig('output/hist.pdf')
plt.clf()
data.plot(kind="scatter",x='longitude',y='latitude',alpha=0.4,
               c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.savefig('output/long_lat.pdf')
plt.clf()
corr_matrix = data.corr()
print corr_matrix["median_house_value"].sort_values(ascending=False)

data.plot(kind="scatter", x="median_income", y="median_house_value",             alpha=0.1)
plt.savefig("housevalue_vs_medianincome.pdf")
plt.clf()
#create 3 new variables
data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"]=data["population"]/data["households"]

corr_matrix = data.corr()
print corr_matrix["median_house_value"].sort_values(ascending=False)
#bedrooms_per_room seems pretty good

#Let us now investigate spliting the dataset randomly and using median_house_price
#and doing stratified sampling


#Now with stratified sampling
#The bulk of the housing prices are under 5.0 ... group housing prices over 5.0 with 5
data["income_cat"] = np.ceil(data["median_income"] / 1.5)
data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True) 
data["income_cat"].hist()
plt.savefig("median_house_price_cat.pdf")
plt.clf()

#create a split for a train and test set with random selection
train_set_rnd,test_set_rnd = train_test_split(data, test_size=0.2,random_state=13)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=13)
for train_index_strat, test_index_strat in split.split(data, data["income_cat"]):
    train_set_strat = data.loc[train_index_strat]
    test_set_strat = data.loc[test_index_strat]
    
train_set_strat["income_cat"].hist()
test_set_strat["income_cat"].hist()
plt.savefig("strat_income_cat.pdf")
plt.clf()
train_set_rnd["income_cat"].hist()
test_set_rnd["income_cat"].hist()
plt.savefig("rnd_income_cat.pdf")
plt.clf()


data.drop("income_cat", axis=1, inplace=True)
data.drop("rooms_per_household",axis=1,inplace=True)
data.drop("bedrooms_per_room",axis=1,inplace=True)
data.drop("population_per_household",axis=1,inplace=True)

for column in data:
#    print column.isnull().sum()
    print column,data[column].isnull().sum()
    #Only total_bedrooms has NAs, 207 of them


#null_columns=data.columns[data.isnull().any()].copy()
#Make a dataframe of rows with NAs

null_data = data[data["total_bedrooms"].isnull()].copy()
notnull_data = data[data["total_bedrooms"].notnull()].copy()

#drop the median_house_value as it is the target, and I
#don't want to train the regressor on it
null_data.drop("median_house_value", axis=1, inplace=True)
notnull_data.drop("median_house_value", axis=1, inplace=True)
#let's try a little imputing on the using regression of some sort
#first we want a train and validation set using data with no NaNs
#
trn_set,val_set = train_test_split(notnull_data, test_size=0.2,random_state=42)
#We also want a copy of the val_set where "total_bedrooms" is NULL
val_set_NA = val_set.copy()


val_set_NA["total_bedrooms"] = np.nan
print trn_set['ocean_proximity'].value_counts()
print val_set_NA['ocean_proximity'].value_counts()

print "trn_set\n",trn_set.head()
print "val_set\n",val_set.head()
print "val_set_NA\n",val_set_NA.head()

NumPipeline_median = Pipeline([
    ('numerical',TypeSelector(np.number)),
    ('imputer',Imputer(strategy='median')),
#    ('std_scaler',StandardScaler())
])
NumPipeline_mode = Pipeline([
    ('numerical',TypeSelector(np.number)),
    ('imputer',Imputer(strategy='most_frequent')),
#    ('std_scaler',StandardScaler())
])
NumPipeline_mean = Pipeline([
    ('numerical',TypeSelector(np.number)),
    ('imputer',Imputer(strategy='mean')),
#    ('std_scaler',StandardScaler())
])

#pipeline 2.
#Select categorical features and convert them to onehotencodered numberical
CatPipeline = Pipeline([
    ('categoricals',TypeSelector('category')),
    ('stringindexer',StringIndexer()),
    ('cat_encoder', OneHotEncoder(sparse=False))
])



FullPipeline_mean = FeatureUnion(transformer_list=[
    ("num_pipeline", NumPipeline_mean),
    ("cat_pipeline", CatPipeline)
])
FullPipeline_mode = FeatureUnion(transformer_list=[
    ("num_pipeline", NumPipeline_mode),
    ("cat_pipeline", CatPipeline)
])
FullPipeline_median = FeatureUnion(transformer_list=[
    ("num_pipeline", NumPipeline_median),
    ("cat_pipeline", CatPipeline)
])

full_output_mean = FullPipeline_mean.fit_transform(trn_set)
full_output_median = FullPipeline_median.fit_transform(trn_set)
full_output_mode = FullPipeline_mode.fit_transform(trn_set)
 

val_set_NA["total_bedrooms"] = np.nan
val_set_NA_mean = FullPipeline_mean.transform(val_set_NA)
val_set_NA["total_bedrooms"] = np.nan
val_set_NA_mode = FullPipeline_mode.transform(val_set_NA)
val_set_NA["total_bedrooms"] = np.nan
val_set_NA_median = FullPipeline_median.transform(val_set_NA)
val_set_NA["total_bedrooms"] = np.nan
#And the version with no nan
val_set_NA_nonull = FullPipeline_median.transform(val_set)

plt.hist(x=np.subtract(val_set_NA_mean[:,4],val_set_NA_nonull[:,4]),bins=40,range=[-2000,2000])
plt.savefig("mean_diff.pdf")
plt.hist(x=np.subtract(val_set_NA_mode[:,4],val_set_NA_nonull[:,4]),bins=40,range=[-2000,2000])
plt.savefig("mode_diff.pdf")
plt.hist(x=np.subtract(val_set_NA_median[:,4],val_set_NA_nonull[:,4]),bins=40,range=[-2000,2000])
plt.savefig("median_diff.pdf")

#Now try linear regression
#Setup training et and target ie. total_bedrooms
trn_set_notarget = trn_set.drop("total_bedrooms",axis=1)
trn_target = trn_set["total_bedrooms"].copy()
val_set_notarget = val_set.drop("total_bedrooms",axis=1)
val_set_target = val_set["total_bedrooms"].copy()

linpipe_num = Pipeline([
    ('numerical',TypeSelector(np.number)),
#    ('imputer',Imputer(strategy='mean')),
    ('std_scaler',StandardScaler())
])


linpipe_cat = Pipeline([
    ('categoricals',TypeSelector('category')),
    ('stringindexer',StringIndexer()),
    ('cat_encoder', OneHotEncoder(sparse=False))
])



linpipe = FeatureUnion(transformer_list=[
    ("num_pipeline", linpipe_num),
    ("cat_pipeline", linpipe_cat)
])

linafter_pipe = linpipe.fit_transform(trn_set_notarget)
linafter_pipe_val_set = linpipe.transform(val_set_notarget)

linreg = LinearRegression()
linreg.fit(linafter_pipe,trn_target)
linregtrn_predict = linreg.predict(linafter_pipe)
linregval_predict = linreg.predict(linafter_pipe_val_set)


print linregval_predict.shape,val_set_target.values.shape
plt.hist(x=np.subtract(linregval_predict,val_set_target.values),bins=40,range=[-2000,2000])
plt.savefig("regression_predict.pdf")
