# for creating the actual scatter plots we want matplotlib.pyplot
import matplotlib.pyplot as plt

# we will use pandas groupby() function to group by species
# when plotting the variables against each other
import pandas as pd

# we will use seaborn pairplot() to present all of the plots
# on one image file
import seaborn as sb

# get the dataframe from the createDataFrame file
from createDataFrame import df

# we will use the os module to create the
# the directory to store the files to
import os

# this function will be called by the main analysis.py file
def createScatterPlots():

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
    plotDir = os.path.join(scriptDir, 'plots/scatterPlots/')
    # we then check if the directory we want to create already
    # exists, and only if it doesn't exist do we invoke the os
    # module's makedirs() method to create it.
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    # there is really no reason when creating scatter plots
    # based on the iris dataset not to distinguish between
    # each of the three variables. When, for example, comparing
    # petal length and petal width, there is no disadvantage
    # to coloring the plots according to the species rather
    # than having a monochromatic plot, and the trends of the
    # variables in the iris data set are clearly related to the
    # species, so by coloring the plots based on the species
    # we can more clearly see the trends and compares how the
    # trends for each species differ. This function then plots
    # the lengths and widths against each other and adding
    # colour to distinguish the species, and also plots the species
    # against all of the lengths and widths

    # we could have used the scatter() function for these plots,
    # but the plot() function works just as well for our use,
    # and is quicker, as demonstrated here:
    # https://pythonmatplotlibtips.blogspot.com/2018/01/compare-pltplot-and-pltscatter-in-speed-python-matplotlib.html


    # these plots plot sepal-length against the other variables in turn,
    # colouring the points according to the species

    # previously I had created custom functions to slice the species-array
    # so that I could plot each species separately. However, I realised
    # afterwards that the pands groupby function can do this, and it is
    # cleaner. For the older version, see createScatterPlots2.py in the
    # filesNotUsed folder

    # groupby works here basically by being called on a dataframe object and taking
    # as a parameter a column name of that dataframe. Then for every unique
    # value in that column, it creates distinct dataframe objects, so that
    # in this case, we get a dataframe object for each of the iris species.
    # It actually returns a dictionary, where the keys are the unique values
    # of the column-parameter inputted, and the values are the dataframes
    # pertaining to that particular key. This means that we can apply the
    # groupby function to the iris dataframe with 'species' inputted as a
    # parameter, and then if we iterate through this with a for loop and
    # and assign the key and value to separate variables, we can use the
    # key variable as the label for the species to be plotted and the
    # value-variable for the actual plotting of the species in question. If
    # we then include these on the same pair of axes, the species will
    # be distinguishable by colour.
    # for documentation see:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
    # and as always, real python is your friend:
    # https://realpython.com/pandas-groupby/
    # this resources provided me the most quickly comprehensible info:
    # https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.html

    # we want to hide the spines to make the plots cleaner
    # (or at least I do!). The docs recommend that to do this
    # you create a subplot and hide the spines from that, see here:
    # https://matplotlib.org/examples/ticks_and_spines/spines_demo.html
    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        # for the third parameter to the plot() function, we use '.'
        # to create a scatter plot, i.e. with dots rather than lines
        plt.plot(group["sepal_length"], group["sepal_width"], '.', label=label)

    # we add a legend to distinguish between the species, as they
    # are included on the same plot. We do not need to include the
    # label parameter, as we have already defined the labels in the
    # plot() function itself.
    # although the legends do not interfere with the points in the plots,
    # it is still safer to make them transparant.
    plt.legend(framealpha=0.5)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Sepal Length vs. Sepal Width")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "sepalLengthSepalWidth.png")
    # close the plot so that the next plot is not
    # superimposed on top of this one
    plt.close()

    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        plt.plot(group["sepal_length"], group["petal_length"], '.', label=label)

    plt.legend(framealpha=0.5)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title("Sepal Length vs. Petal Length")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "sepalLengthPetalLength.png")
    plt.close()

    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        plt.plot(group["sepal_length"], group["petal_width"], '.', label=label)

    plt.legend(framealpha=0.5)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Sepal Length vs. Petal Width")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "sepalLengthPetalWidth.png")
    plt.close()

    # we want to plot the above plot with the axes reversed
    # to compare with the LDA plot later, as sepal length vs.
    # petal width arguably provides the best off-the-cuff
    # differentiation between the species
    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        plt.plot(group["petal_width"], group["sepal_length"], '.', label=label)

    plt.legend(framealpha=0.5)
    plt.xlabel("Petal Width (cm)")
    plt.ylabel("Sepal Length (cm)")
    plt.title("Petal Width vs. Sepal Length")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "petalWidthSepalLength.png")
    plt.close()







    # remaining sepal-width plots
    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        plt.plot(group["sepal_width"], group["sepal_length"], '.', label=label)


    plt.legend(framealpha=0.5)
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title("Sepal Width vs. Sepal Length")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "sepalWidthSepalLength.png")
    plt.close()

    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        plt.plot(group["sepal_width"], group["petal_width"], '.', label=label)

    plt.legend(framealpha=0.5)
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Sepal Width vs. Petal Width")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "sepalWidthPetalWidth.png")
    plt.close()

    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        plt.plot(group["petal_length"], group["petal_width"], '.', label=label)

    plt.legend(framealpha=0.5)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Petal Length vs. Petal Width")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "petalLengthPetalWidth.png")
    plt.close()

    # it is hard to resist using seaborn to create a matrix of
    # the above plots on a single image file
    # the following code uses the pairplot() function, where
    # the first parameter is the pandas dataframe, and the
    # second refers to the variables which are given distinct,
    # wait for it, HUES. There are many more optional variables,
    # but the default values work fine here
    # for information of seaborn.pairplot, see the documentation:
    # https://seaborn.pydata.org/generated/seaborn.pairplot.html
    # and also here for a good introduction:
    # https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
    sb.pairplot(data=df, hue="species")
    # we use matplotlit.pyplot to save the image to a file
    plt.savefig(plotDir + "scatterMatrix.png")
    plt.close()
    # we use a raw string here to accurately depict the file path on
    # Windows systems, i.e. with '\' as the path separator.
    # see here for the use of raw string:
    # https://stackoverflow.com/questions/4415259/convert-regular-python-string-to-raw-string
    print(r"Scatter plots of the Iris dataset have been created and saved to the 'plots\scatterPlots' directory.")

# if this is run as a script, we should call the
# createHistograms() function, as that is what
# someone would expect to happen should they choose
# to run this rile on its own
if __name__ == '__main__':
    createScatterPlots()
