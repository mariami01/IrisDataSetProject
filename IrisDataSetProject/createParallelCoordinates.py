# get the dataframe from the createDataFrame file
from createDataFrame import df
# we will use pyplot to save the plot to an image file
import matplotlib.pyplot as plt
# we will use pandas to create the parallel coordinates plot
import pandas as pd
# we will use the os module to create the
# the directory to store the files to
import os



def createParallelCoordinates():

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
    plotDir = os.path.join(scriptDir, 'plots/parallelCoordinates/')
    # we then check if the directory we want to create already
    # exists, and only if it doesn't exist do we invoke the os
    # module's makedirs() method to create it.
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    # the pandas docs give a good explanation of how to
    # create a parralel coordinates plot:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.parallel_coordinates.html
    # basically, the first parameter is the pandas dataframe
    # object, the second parameter is the column name that
    # want to plot against the other columns, and whose
    # different variables will be distinguishable by color
    # in the plot. All the other parameters are optional.
    # Even the legend and the colouring can be left to default.
    pd.plotting.parallel_coordinates(
            df, 'species')
    # we do want to add a title, however:
    plt.title("Parallel Coordinates Plot")
    # as well as including the unit on the y axis:
    plt.ylabel("Centimeters")
    # concatenate the directory created above with the name
    # of the file and save the plot to this path.
    plt.savefig(plotDir + "parallelCoordinates.png")

    # we use a raw string here to accurately depict the file path on
    # Windows systems, i.e. with '\' as the path separator.
    # see here for the use of raw string:
    # https://stackoverflow.com/questions/4415259/convert-regular-python-string-to-raw-string
    print(r"A parallel coorindates plot of the Iris dataset has been created and saved to the 'plots\parallelCoordinates' directory.")


if __name__ == '__main__':
    createParallelCoordinates()
