import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt

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
