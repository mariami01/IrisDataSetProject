import pandas as pd

# import the dataframe of the dataset used in this project
from createDataFrame import df

def verifyIntegrityOfDataset():

    # get the url of known good source for the dataset
    # here I use a popular version of the dataset on Git Hub Gist
    # note that one must use the raw dataset
    url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv"

    # create a dataframe object of the verified dataset
    verifiedDataSet = pd.read_csv(url)

    # check if the verified dataset and the one used in the project are equal.
    # Note that we must use the DataFrame.equals() method
    # rather than the equality operator - == - as DataFrames
    # are ambiguous in respect of boolean values. See here:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.equals.html
    assert verifiedDataSet.equals(df), "This dataset would appear to be corrupted."
    print("The integrity of the data set has been verified against a known good source.")


if __name__ == '__main__':
    verifyIntegrityOfDataset()
