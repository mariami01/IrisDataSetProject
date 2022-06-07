# we want numpy for creating numpy arrays with linspace
# method. Linspace can easily create arrays with steps
# of floating point magnitudes, which will be helpful
# for specifying the bins for the histograms.
import numpy as np
# for creating the actual histograms we want matplotlib.pyplot
import matplotlib.pyplot as plt
# get the dataframe from the createDataFrame file
from createDataFrame import df
# we will use pandas groupby() function in the file, but
# interestingly we don't actually need to import pandas here
# because we will by using the dataframe imported above to
# call the groupby function.

# we will use the os module to create the
# the directory to store the files to
import os

# this function will be called by the main analysis file
def createHistograms():

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
    plotDir = os.path.join(scriptDir, 'plots/histograms/')
    # we then check if the directory we want to create already
    # exists, and only if it doesn't exist do we invoke the os
    # module's makedirs() method to create it.
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)


    # originally I created histograms for all of the variables
    # (see createHistograms1.py in filesNotUsed folder).
    # However, as histograms only plot one variable, they are
    # really only useful when compared against each other.
    # Thus, I instead created one images with histograms for
    # each of the variables regardless of species. Next I created
    # plots for the variables (this time saved to separate images)
    # by plotting the variables separately for each species, but
    # including each of the three plots for each variable on the same
    # axis. This allows differences between the species to be more
    # easily compared. Finally I incorporated all of the four plots
    # created in the last step into a single image, for even
    # easier visual comparison.

    # to incorporate multiple axes on the one image, use the subplot
    # method. The first paramater defines the number of axes
    # vertically, the next horizontally, and the third parameter (an integer)
    # defines what axes will be plotted next, where '1' refers to the
    # top left axis, '2' refers to the axis to the right of '1' (or below
    # if there is no plot to the right) and so on until the last (bottom right) axis.
    # see the documentation here:
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html

    # In the case of creating histograms for each of the
    # variables independent of species, I decided not to vary
    # the bins across the histograms. While this is not
    # ideal in the sense that there will be a lot of "white
    # space" on some of the histograms, I think this is made up for
    # by the ease with which one can compare the histograms, i.e. one
    # does not have to factor in the range of the bins but can compare
    # the variables and species purely by looking at the plots.
    # 8 is the upper bound (virginica sepal length) and 0 is the lower
    # (setosa petal width). While bins of 0.25 width are probably the
    # most visually appealing, the setosa-petal values are clumped
    # together so much that bins with a width of 0.25 really aren"t
    # granular enough. Bins of 0.125 have been used instead.

    plt.subplot(2,2,1)
    # linspace function was mentioned in lecture videos
    plt.hist(df["sepal_length"], bins=np.linspace(0,8,64))
    # we don"t want to label the x axis here, as it would
    # overlap with the plot under it
    # plt.xlabel("Length (cm)")
    # include the label for the y axis
    plt.ylabel("Occurences")
    # we want the xticks and yticks to be the same across
    # all the subplots. The xticks will be the same because
    # the bins are the same, the yticks need to be explicitly
    # defined. We don"t want things to get too messy, so ticks
    # of 0, 10 and 20 should suffice.
    plt.yticks([0,10,20])

    plt.grid()
    # give the plot its title
    plt.title("Overall Sepal Length")

    # as we are creating subplots all on the one file, we do not
    # save it yet, nor close the plot

    plt.subplot(2,2,2)
    plt.hist(df["sepal_width"], bins=np.linspace(0,8,64))
    # we don"t either of the labels here, and they would overlap with
    # the other plots" axes anyway
    # plt.xlabel("Width (cm)")
    # plt.ylabel("Occurences")
    plt.yticks([0,10,20])
    # grids go well with histograms. I think due to histograms' blocky
    # nature, the grid isn't as distracting as in the case of scatter
    # plots for example.
    plt.grid()
    plt.title("Overall Sepal Width")

    plt.subplot(2,2,3)
    plt.hist(df["petal_length"], bins=np.linspace(0,8,64))
    plt.xlabel("Length (cm)")
    plt.ylabel("Occurences")
    plt.yticks([0,10,20])
    plt.grid()
    plt.title("Overall Petal Length")

    plt.subplot(2,2,4)
    plt.hist(df["petal_width"], bins=np.linspace(0,8,64))
    plt.xlabel("Width (cm)")
    # plt.ylabel("Occurences")
    plt.yticks([0,10,20])
    plt.grid()
    plt.title("Overall Petal Width")

    # To avoid the titles of the bottom plots and the x axis labels
    # of the upper plots from overlapping, call the tigh_layout()
    # function. For documentation, see here:
    # https://matplotlib.org/3.1.3/tutorials/intermediate/tight_layout_guide.html
    plt.tight_layout()
    # save it to a file in the plots/histogram folder
    plt.savefig(plotDir + "allSpeciesHistograms.png")
    # and close the plot so that the next plot is not
    # superimposed on top of this one
    plt.close()


    # when plotting multiple plots on the same axes, the easiest
    # way is to use the groupby function in pandas. The use of this
    # is explained in more detail in the createScatterPlots.py file.
    # Here, it basically allows us to create separate histograms
    # for each of the species, and then show them on the same
    # axes. Because we have a lot of data on each axis now,
    # we want to make the legend semi-transpart with the
    # alphaframe keyword argument, and the plots themselves
    # semi-transparent with the alpha keword argument of the
    # hist function. Because we are concerned here with comparing
    # the species against each other on the same axes, there is less
    # need to have the xticks and yticks and bins the same across
    # all axes, so we can leave the ticks unspecified, and specify
    # 10 bins for each plot.

    # for alphaframe keyword argument of legend() see here:
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.legend.html
    # for alpha see here:
    # http://www.learningaboutelectronics.com/Articles/How-to-change-the-transparency-of-a-graph-plot-in-matplotlib-with-Python.php

    # one can actually call the hist() method with the result of a groupby() call,
    # assign the list of values grouped-by to the labels keyword argument,
    # and the hist will create plots for each of the groups! This
    # doesn't seem possible when making scatter plots, perhaps because
    # plot() in that case requires two arguments, and so one would have to both
    # call the plot() method with one groupby() result, and then include
    # another groupby() result as the first parameter to the plot() method,
    # which doesn't intuitively make sense, and apparently doesn't
    # make sense to pyplot, as I haven't been able to make that work.
    # Note that it is possible to create multiple plots with the one call
    # to hist() because the array-like parameter that is plotted can
    # actually be 2D, in which each column is a dataset, as explained here:
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.hist.html
    # there is also a very nice explanation here
    # https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm

    # However, I have not chosen to do this, but instead have used a for loop
    # to iterate through the groups, each time labeling the resulting
    # plot appropriately. This allows one to more intuitively (I think)
    # superimpose the plots onto each other and create a legend.

    # to get a list of the species, we can call the unique() method
    # on the array-like object df.species. For more on unique() see:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    labels = df.species.unique()

    # Sepal Length for each species
    for label, group in df.groupby("species"):
        plt.hist(group["sepal_length"], bins=10, alpha=0.5, label=label)
    plt.title("Sepal Length")
    plt.xlabel("Length (cm)")
    plt.ylabel("Occurences")
    plt.grid()
    plt.legend()
    plt.savefig(plotDir + "sepalLength.png")
    plt.close()

    # Sepal Width for each species
    for label, group in df.groupby("species"):
        plt.hist(group["sepal_width"], bins=10, alpha=0.5, label=label)
    plt.title("Sepal Width")
    plt.xlabel("Width (cm)")
    plt.ylabel("Occurences")
    plt.grid()
    plt.legend()
    plt.savefig(plotDir + "sepalWidth.png")
    plt.close()

    # Petal Length for each species
    for label, group in df.groupby("species"):
        plt.hist(group["petal_length"], bins=10, alpha=0.5, label=label)
    plt.title("Petal Length")
    plt.xlabel("Length (cm)")
    plt.ylabel("Occurences")
    plt.grid()
    plt.legend()
    plt.savefig(plotDir + "petalLength.png")
    plt.close()

    # Petal Width for each species
    for label, group in df.groupby("species"):
        plt.hist(group["petal_width"], bins=10, alpha=0.5, label=label)
    plt.title("Petal Width")
    plt.xlabel("Width (cm)")
    plt.ylabel("Occurences")
    plt.grid()
    plt.legend()
    plt.savefig(plotDir + "petalWidth.png")
    plt.close()

    # now we include the above four axes on the one image with subplotting
    plt.subplot(2,2,1)
    for label, group in df.groupby("species"):
        plt.hist(group["sepal_length"], bins=10, alpha=0.5, label=label)
    plt.title("SepalLength")
    #plt.xlabel("Length (cm)")
    plt.ylabel("Occurences")
    plt.legend(framealpha=0.5)
    plt.grid()

    plt.subplot(2,2,2)
    for label, group in df.groupby("species"):
        plt.hist(group["sepal_width"], bins=10, alpha=0.5, label=label)
    plt.title("SepalWidth")
    #plt.xlabel("Width (cm)")
    #plt.ylabel("Occurences")
    plt.legend(framealpha=0.5)
    plt.grid()

    plt.subplot(2,2,3)
    for label, group in df.groupby("species"):
        plt.hist(group["petal_length"], bins=10, alpha=0.5, label=label)
    plt.title("PetalLength")
    plt.xlabel("Length (cm)")
    plt.ylabel("Occurences")
    plt.legend(framealpha=0.5)
    plt.grid()

    plt.subplot(2,2,4)
    for label, group in df.groupby("species"):
        plt.hist(group["petal_width"], bins=10, alpha=0.5, label=label)
    plt.title("PetalWidth")
    plt.xlabel("Width (cm)")
    #plt.ylabel("Occurences")
    plt.legend(framealpha=0.5)

    plt.grid()
    plt.tight_layout()

    plt.savefig(plotDir + "overallHistograms.png")
    plt.close()

    # we use a raw string here to accurately depict the file path on
    # Windows systems, i.e. with '\' as the path separator.
    # see here for the use of raw string:
    # https://stackoverflow.com/questions/4415259/convert-regular-python-string-to-raw-string
    print(r"Histograms of the Iris dataset have have been created and saved to the 'plots\histograms' directory.")



# if this is run as a script, we should call the
# createHistograms() function, as that is what
# someone would expect to happen should they choose
# to run this rile on its own
if __name__ == '__main__':
    createHistograms()
