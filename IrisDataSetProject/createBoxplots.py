# get the dataframe from the createDataFrame file
from createDataFrame import df
import matplotlib.pyplot as plt
# we will use seaborn for plotting here. While pandas
# can be used for boxplots, seaborn is better.
import seaborn as sb
# there"s a very nice introduction to creating boxplots
# with seaborn here. It also demonstrates how to overlay
# over plots on top of box plots for greater clarity.
# Swarm plot is particularly effective when overlayed
# on a boxplot.

# we will use the os module to create the
# the directory to store the files to
import os

def createBoxPlots():

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
    plotDir = os.path.join(scriptDir, 'plots/boxPlots/')
    # we then check if the directory we want to create already
    # exists, and only if it doesn't exist do we invoke the os
    # module's makedirs() method to create it.
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)




    # because only an axes object can call the set_title method,
    # we assign the boxplot to a variable, and then use
    # that to call the set_title method
    plot = sb.boxplot(y="sepal_length", x="species", data=df, width=0.5)
    plot.set_title("Species vs. Sepal Length")
    # overlay a swarm plot, specifying the color of the dots
    # as black, and making them somewhat transparent.
    plot = sb.swarmplot(y="sepal_length", x="species", data=df, color="black", alpha=0.75)
    # again, seaborn is not as user-friendly as pyplot, and
    # we must use the plot's 'figure' object to call savefig()
    plt.grid()
    plot.figure.savefig(plotDir + "sepalLength.png")
    # apparently, pyplot should be used to close seaborn plots:
    # https://stackoverflow.com/questions/57533954/how-to-close-seaborn-plots
    plt.close()

    plot = sb.boxplot(y="sepal_width", x="species", data=df, width=0.5)
    plot.set_title("Species vs. Sepal Width")
    plot = sb.swarmplot(y="sepal_width", x="species", data=df, color="black", alpha=0.75)
    plt.grid()
    plot.figure.savefig(plotDir + "sepalWidth.png")
    plt.close()

    plot = sb.boxplot(y="petal_length", x="species", data=df, width=0.5)
    plot.set_title("Species vs. Petal Length")
    plot = sb.swarmplot(y="petal_length", x="species", data=df, color="black", alpha=0.75)
    plt.grid()
    plot.figure.savefig(plotDir + "petalLength.png")
    plt.close()

    plot = sb.boxplot(y="petal_width", x="species", data=df, width=0.5)
    plot.set_title("Species vs. Petal Width")
    plot = sb.swarmplot(y="petal_width", x="species", data=df, color="black", alpha=0.75)
    plt.grid()
    plot.figure.savefig(plotDir + "petalWidth.png")
    plt.close()

    # as thee setosa petal length and width are closely clustered,
    # their boxplots are not very revealing when plotted
    # alongside the other species'. For this reason we now plot
    # them both separately.

    # we use a raw string here to accurately depict the file path on
    # Windows systems, i.e. with '\' as the path separator.
    # see here for the use of raw string:
    # https://stackoverflow.com/questions/4415259/convert-regular-python-string-to-raw-string
    print(r"Boxplots of the Iris dataset overlayed with swarm plots have been created and saved to the 'plots\boxPlots' directory.")

# if this is run as a script, we should call the
# createBoxplots() function, as that is what
# someone would expect to happen should they choose
# to run this rile on its own
if __name__ == '__main__':
    createBoxPlots()
