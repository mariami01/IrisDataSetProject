# for creating the actual scatter plots we want matplotlib.pyplot
import matplotlib.pyplot as plt

import seaborn as sb

# get the dataframe from the createDataFrame file
from createDataFrame import df
# import the indices that will allow us create separate plots
# for each of the species (for more on how this is achieved
# see comments in createDataFrame file)
from createDataFrame import versicolor
from createDataFrame import virginica

# this function will be called by the main analysis file
def createScatterPlots():

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

    # species vs other-variables plots

    # for the third parameter to the plot() function, we use '.'
    # to create a scatter plot, i.e. with dots rather than lines
    plt.plot(df["species"], df["sepal_length"], ".")
    # x label
    plt.xlabel("Species")
    # y label
    plt.ylabel("Sepal Length (cm)")
    # title
    plt.title("Species vs. Sepal Length")
    # save in plots/scatterPlots folder
    plt.savefig("plots/scatterPlots/speciesSepalLength.png")
    # close the plot so that the next plot is not
    # superimposed on top of this one
    plt.close()

    plt.plot(df["species"], df["sepal_width"], ".")
    plt.xlabel("Species")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Species vs. Sepal Width")
    plt.savefig("plots/scatterPlots/speciesSepalWidth.png")
    plt.close()

    plt.plot(df["species"], df["petal_length"], ".")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.title("Species vs. Petal Length")
    plt.savefig("plots/scatterPlots/speciesPetalLength.png")
    plt.close()

    plt.plot(df["species"], df["petal_width"], ".")
    plt.xlabel("Species")
    plt.ylabel("PetalWidth (cm)")
    plt.title("Species vs. PetalWidth")
    plt.savefig("plots/scatterPlots/speciesPetalWidth.png")
    plt.close()

    # Sepal-length plots
    import pandas as pd
    # we add a label here to distinguish between each of the species
    plt.plot(df["sepal_length"][:versicolor], df["sepal_width"][:versicolor], '.', label="setosa")
    plt.plot(df["sepal_length"][versicolor:virginica], df["sepal_width"][versicolor:virginica], '.', label="versicolor")
    plt.plot(df["sepal_length"][virginica:], df["sepal_width"][virginica:], '.', label="virginica")

    # we add a legend to distinguish between the species, as they
    # are included on the same plot
    plt.legend()
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Sepal Length vs. Sepal Width")
    plt.savefig("plots/scatterPlots/sepalLengthSepalWidth.png")
    plt.close()

    plt.plot(df["sepal_length"][:versicolor], df["petal_length"][:versicolor], '.', label="setosa")
    plt.plot(df["sepal_length"][versicolor:virginica], df["petal_length"][versicolor:virginica], '.', label="versicolor")
    plt.plot(df["sepal_length"][virginica:], df["petal_length"][virginica:], '.', label="virginica")

    plt.legend()
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title("Sepal Length vs. Petal Length")
    plt.savefig("plots/scatterPlots/sepalLengthPetalLength.png")
    plt.close()

    plt.plot(df["sepal_length"][:versicolor], df["petal_width"][:versicolor], '.', label="setosa")
    plt.plot(df["sepal_length"][versicolor:virginica], df["petal_width"][versicolor:virginica], '.', label="versicolor")
    plt.plot(df["sepal_length"][virginica:], df["petal_width"][virginica:], '.', label="virginica")

    plt.legend()
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Sepal Length vs. Petal Width")
    plt.savefig("plots/scatterPlots/sepalLengthPetalWidth.png")
    plt.close()

    # remaining sepal-width plots

    plt.plot(df["sepal_width"][:versicolor], df["petal_length"][:versicolor], '.', label="setosa")
    plt.plot(df["sepal_width"][versicolor:virginica], df["petal_length"][versicolor:virginica], '.', label="versicolor")
    plt.plot(df["sepal_width"][virginica:], df["petal_length"][virginica:], '.', label="virginica")

    plt.legend()
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title("Sepal Width vs. Sepal Length")
    plt.savefig("plots/scatterPlots/sepalWidthPetalLength.png")
    plt.close()

    plt.plot(df["sepal_width"][:versicolor], df["petal_width"][:versicolor], '.', label="setosa")
    plt.plot(df["sepal_width"][versicolor:virginica], df["petal_width"][versicolor:virginica], '.', label="versicolor")
    plt.plot(df["sepal_width"][virginica:], df["petal_width"][virginica:], '.', label="virginica")

    plt.legend()
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Sepal Width vs. Petal Width")
    plt.savefig("plots/scatterPlots/sepalWidthPetalWidth.png")
    plt.close()

    # remaining petal-length plots

    plt.plot(df["petal_length"][:versicolor], df["petal_width"][:versicolor], '.', label="setosa")
    plt.plot(df["petal_length"][versicolor:virginica], df["petal_width"][versicolor:virginica], '.', label="versicolor")
    plt.plot(df["petal_length"][virginica:], df["petal_width"][virginica:], '.', label="virginica")


    plt.legend()
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Petal Length vs. Petal Width")
    plt.savefig("plots/scatterPlots/petalLengthPetalWidth.png")
    plt.close()


    # it is hard to resist using seaborn to create a matrix of
    # the above plots on a single image file
    # the following code uses the pairplot() function, where
    # the first parameter is the pandas dataframe, and the
    # second refers to the variables which are given distinct,
    # wait for it, HUES
    sb.pairplot(data=df, hue="species")
    # we use matplotlit.pyplot to save the image to a file
    plt.savefig("plots/scatterPlots/scatterMatrix.png")
