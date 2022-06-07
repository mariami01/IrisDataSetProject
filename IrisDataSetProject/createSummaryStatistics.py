from createDataFrame import df
# we will use the os module to create the
# the directory to store the files to
import os

def createSummaryStatistics():

    # I adapted the code for creating a directory
    # if it doesn't already exist from here:
    # https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
    # first we get the absolute path of the current directory
    # by calling the dirname() method on the os module's path
    # field with a parameter of the current filename
    # (represented by the 'dunder' or 'magic' __file__
    # variable that is built into Python).
    scriptDir = os.path.dirname(__file__)
    # we then create the path of the new directory by calling
    # the join() method on the os module's path field with two
    # parameter, the first being the current directory's path, and the
    # second being the subdirectory we want to add to it.
    plotDir = os.path.join(scriptDir, 'summaryStatistics/')
    # we then check if the directory we want to create already
    # exists, and only if it doesn't exist do we invoke the os
    # module's makedirs() method to create it.
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    # see here for an explanation of 'loc':
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
    # we basically want to slice the dataframe according to the species,
    # we can do this with loc[]'s filtering feature, whereby a boolean
    # expression can be used as follows: loc[boolean expression]
    # the rows will be returned in which the boolean expression
    # evaluates to true
    # We don't really want the overal summary statistics, but rather
    # those for each of the species
    setosaSummary = df.loc[df['species'] == "setosa"].describe()
    versicolorSummary = df.loc[df['species'] == "versicolor"].describe()
    virginicaSummary = df.loc[df['species'] == "virginica"].describe()
    # write them to csv files
    setosaSummary.to_csv(plotDir + "setosaSummaryStats.csv")
    versicolorSummary.to_csv(plotDir + "versicolorSummaryStats.csv")
    virginicaSummary.to_csv(plotDir + "virginicaSummaryStats.csv")

    print("Summary statistics for each species have been written to csv files in the 'summaryStatistics' folder.")




if __name__ == '__main__':
    createSummaryStatistics()
