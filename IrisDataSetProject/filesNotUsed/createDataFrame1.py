import pandas as pd

# read in the csv file and assign it to a variable
# (it is now a pandas dataframe object)
df = pd.read_csv("irisDataSet.csv")

# these variables will store the indices of the first
# versicolor and virginicas
versicolor = 0
virginica = 0

# this function finds the indexes of the first versicolor
# and virginica species. The idea is that this will allow
# one to plot each species separately by using these indices
# to slice the dataframe. This way, when plotting, for example,
# petal length vs petal width, one can plot this for each
# species separately, and then include all the plots on the
# same pair of axes, and the points representing each species
# will have a separate color. Additionally, because the species
# are plotted separately, each can be labelled so that they
# are distinguishable in the legend, i.e. the legend describes
# the color used to plot each of the species. Also, histograms
# for each of the species' variables can be plotted this way. The
# reason for plotting the species separately is given in the README.
def findIndexesForSlicing(df):
    global versicolor
    global virginica
    # iterate through the data, and once the first versicolor is arrived
    # at, assign the index to the versicolor variable and break. The same
    # process applies below for virginica
    for i in range(len(df)):
        if df[i] =='versicolor':
            versicolor = i
            break
    for i in range(len(df)):
        if df[i] == 'virginica':
            virginica = i
            break

# it makes more sense to call this function now, so that instead of
# having to import the function into the other files and then run it,
# we can just import the variables directly from this file
findIndexesForSlicing(df["species"])


# it is worth noting here that the above method was the second and preferred
# method for making it possible to plot two variables on the same axes,
# such as petal lenth and petal width, while having different colours to
# represent the different axes. The first method I came upon I found here:
# https://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
# the idea is to utilise the 'c' keyword argument that can be included in
# calls to the plot() function of matplotlib.pyplot. This keyword determines
# the color of the points to be plotted. It can take in an array of the same
# size as the arrays to be plotted, in which case every point can have
# a different color if so desired. The idea here is that an array could be used
# where the indexes of each species mapped to a particular color in the array, e.g.
# if versicolors are from index 50-100 in the data set then indexes 50-100
# in the 'c' array could refer to green. The problem here is that all of the
# species are plotted together, so it becomes more difficult to include a
# legend where each species is referred to individually. This is why this method
# was not chosen. The following function, adapted from the above website,
# maps colors as a list from the input list of x variables: the logic is
# virtually identical to the function above, except a list is generated
# rather than key indices being extracted.
''' def pltcolor(lst):
    cols=[]
    for l in lst:
        if l=='setosa':
            cols.append('red')
        elif l=='versicolor':
            cols.append('blue')
        else:
            cols.append('green')
    return cols'''
