# we want numpy for creating numpy arrays with linspace
# method. Linspace can easily create arrays with steps
# of floating point magnitudes, which will be needed
# to specify the bins for the histograms.
import numpy as np
# for creating the actual histograms we want matplotlib.pyplot
import matplotlib.pyplot as plt
# get the dataframe from the createDataFrame file
from createDataFrame import df
# import the indices that will allow us create separate plots
# for each of the species (for more on how this is achieved
# see comments in createDataFrame file)
from createDataFrame import versicolor
from createDataFrame import virginica

# this function will be called by the main analysis file
def createHistograms():

    # first create histograms for each variable except species
    # where all species are lumped together

    # the steps for creating the histograms are the same for each
    # variable plotted, so I will annotate the process only once.
    # First plot the histogram with the appropriate array. The second
    # paramater here specifies the width and range of the bins. I have
    # decided not to vary this across the histograms. While this
    # is not ideal in the sense that there will be a lot of 'white
    # space' on most of the histograms, I think this is made up for
    # by the ease with which one can compare the histograms, i.e. one
    # does not have to factor in the range of the bins but can compare
    # the variables and species purely by looking at the plots.
    # 8 is the upper bound (virginica sepal length) and 0 is the lower
    # (setosa petal width). While bins of 0.25 width are probably the
    # most visually appealing, the setosa-petal values are clumped
    # together so much that histograms with a width of 0.5 really aren't
    # granular enough. Bins of 0.125 have been used instead.
    # but the setosa petal lengths and widths

    plt.subplot(2,2,1)

    plt.hist(df["sepal_length"], bins=np.linspace(0,8,64))
    # we don't want to label the x axis here, as it would
    # overlap with the plot under it
    # plt.xlabel("Length (cm)")
    # include the label for the y axis
    plt.ylabel("Occurences")
    # we want the xticks and yticks to be the same across
    # all the subplots. The xticks will be the same because
    # the bins are the same, the yticks need to be explicitly
    # defined. We don't want things to get too messy, so ticks
    # of 0, 10 and 20 should suffice.
    plt.yticks([0,10,20])

    # give the plot its title
    plt.title("Overall Sepal Length")

    # as we are creating subplots all on the one file, we do not
    # save it yet, nor close the plot

    plt.subplot(2,2,2)
    plt.hist(df["sepal_width"], bins=np.linspace(0,8,64))
    # we don't either of the labels here, and they would overlap with
    # the other plots' axes anyway
    # plt.xlabel("Width (cm)")
    # plt.ylabel("Occurences")
    plt.yticks([0,10,20])
    plt.title("Overall Sepal Width")

    plt.subplot(2,2,3)
    plt.hist(df["petal_length"], bins=np.linspace(0,8,64))
    plt.xlabel("Length (cm)")
    plt.ylabel("Occurences")
    plt.yticks([0,10,20])
    plt.title("Overall Petal Length")

    plt.subplot(2,2,4)
    plt.hist(df["petal_width"], bins=np.linspace(0,8,64))
    plt.xlabel("Width (cm)")
    # plt.ylabel("Occurences")
    plt.yticks([0,10,20])
    plt.title("Overall Petal Width")

    # To avoid the titles of the bottom plots and the x axis labels
    # of the upper plots from overlapping.
    plt.tight_layout()
    # save it to a file in the plots/histogram folder
    plt.savefig("plots/histograms/allSpeciesHistograms.png")
    # and close the plot so that the next plot is not
    # superimposed on top of this one
    plt.close()




    # setosa
    plt.hist(df["sepal_length"][:versicolor], bins=np.linspace(0,8,64))
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Occurences")
    plt.title("Setosa Sepal Length")
    plt.savefig("plots/histograms/setosaSepalLength.png")
    plt.close()

    plt.hist(df["sepal_width"][:versicolor], bins=np.linspace(0,8,64))
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Occurences")
    plt.title("Setosa Sepal Width")
    plt.savefig("plots/histograms/setosaSepalWidth.png")
    plt.close()

    plt.hist(df["petal_length"][:versicolor], bins=np.linspace(0,8,64))
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Occurences")
    plt.title("Setosa Petal Length")
    plt.savefig("plots/histograms/setosaPetalLength.png")
    plt.close()

    plt.hist(df["petal_width"][:versicolor], bins=np.linspace(0,8,64))
    plt.xlabel("Petal Width (cm)")
    plt.ylabel("Occurences")
    plt.title("Setosa Petal Width")
    plt.savefig("plots/histograms/setosaPetalWidth.png")
    plt.close()

    # versicolor
    plt.hist(df["sepal_length"][versicolor:virginica], bins=np.linspace(0,8,64))
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Occurences")
    plt.title("Versicolor Sepal Length")
    plt.savefig("plots/histograms/versicolorSepalLength.png")
    plt.close()

    plt.hist(df["sepal_width"][versicolor:virginica], bins=np.linspace(0,8,64))
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Occurences")
    plt.title("Versicolor Sepal Width")
    plt.savefig("plots/histograms/versicolorSepalWidth.png")
    plt.close()

    plt.hist(df["petal_length"][versicolor:virginica], bins=np.linspace(0,8,64))
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Occurences")
    plt.title("Versicolor Petal Length")
    plt.savefig("plots/histograms/versicolorPetalLength.png")
    plt.close()

    plt.hist(df["petal_width"][versicolor:virginica], bins=np.linspace(0,8,64))
    plt.xlabel("Petal Width (cm)")
    plt.ylabel("Occurences")
    plt.title("Versicolor Petal Width")
    plt.savefig("plots/histograms/versicolorPetalWidth.png")
    plt.close()

    # virginica
    plt.hist(df["sepal_length"][virginica:], bins=np.linspace(0,8,64))
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Occurences")
    plt.title("Virginica Sepal Length")
    plt.savefig("plots/histograms/virginicaSepalLength.png")
    plt.close()

    plt.hist(df["sepal_width"][virginica:], bins=np.linspace(0,8,64))
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Occurences")
    plt.title("Virginica Sepal Width")
    plt.savefig("plots/histograms/virginicaSepalWidth.png")
    plt.close()

    plt.hist(df["petal_length"][virginica:], bins=np.linspace(0,8,64))
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Occurences")
    plt.title("Virginica Petal Length")
    plt.savefig("plots/histograms/virginicaPetalLength.png")
    plt.close()

    plt.hist(df["petal_width"][virginica:], bins=np.linspace(0,8,64))
    plt.xlabel("Petal Width (cm)")
    plt.ylabel("Occurences")
    plt.title("Virginica Petal Width")
    plt.savefig("plots/histograms/virginicaPetalWidth.png")
    plt.close()

