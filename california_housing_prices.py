import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import TransformerMixin, BaseEstimator, clone

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
data = load_data_csv("input","housing.csv")

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

#Now we will drop the income_cat variables
for set in (train_set_rnd, test_set_rnd,train_set_strat,test_set_strat):
    set.drop(["income_cat"], axis=1, inplace=True) 

#We will now setup the pipeline to deal with the daa and
#prepare it for use

#What do we want to do?
#Deal with the categoical variable ocean_proximity by encodeing
#Impute missing values (if any)
#Scale data
#Aplly function, such as log, to data with long tails

#drop the target
train_set = train_set_strat.drop("median_house_value",axis=1)
#save the target
train_target = train_set_strat["median_house_value"].copy()
