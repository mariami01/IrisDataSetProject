# get the dataframe from the createDataFrame file
from createDataFrame import df

# for creating the actual scatter plots we want matplotlib.pyplot
import matplotlib.pyplot as plt

# we only need access to the 'LinearDiscriminantAnalysis'
# and 'PrincipalComponentAnalysis'
# functions from sklearn.discriminat_analysis
# and sklearn.decomposition respectively
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

# we will use the os module to create the
# the directory to store the files to
import os

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
plotDir = os.path.join(scriptDir, 'plots/dimensionalityReduction/')
# we then check if the directory we want to create already
# exists, and only if it doesn't exist do we invoke the os
# module's makedirs() method to create it.
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)



def createLDAScatterPlot():

    # the code for this is adapted from here:
    # # https://sebastianraschka.com/Articles/2014_python_lda.html

    # we want to split the dataset into the variables proper
    # and the classes, i.e. the species. dataframe.iloc[]
    # can do this, which takes two parameters, the first slicing
    # the rows, the second the columns. Here we want all the rows,
    # and the first four columns. In the second case we only want the
    # fifth column
    X = df.iloc[:,0:4]
    Y = df.iloc[:,4]

    # we want to plot this on a 2D plane, so we want to use
    # two linear discriminants to perform the LDA, which two
    # discriminants will then become the two dimensions that we plot
    sklearn_lda = LDA(n_components=2)

    # there is a very nice explanation of how the fit() and transform()
    # methods work here:
    # https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn
    # basically, fit() takes in the training data as a parameter,
    # is called with one of sklearn's dimension-reducing algorithms,
    # and then creates the learning model that will, once the transform()
    # method is called with training/test data, translate each datum's
    # variable values into the discriminant values by which, in the case
    # of LDA, the datasets classes can best be discriminated by.

    # ordinarily we would split the dataset into training data
    # and testing data, but because here we want to visualize the
    # lda in a plot, we can just use the whole dataset
    X_lda_sklearn = sklearn_lda.fit_transform(X, Y)
    # we now just plot these values as normal, using a scatter plot
    # so that the individual data points are distinguishable
    ax = plt.subplot(111)
    # The zip() function returns a zip object, which is an iterator
    # of tuples where the first item in each passed iterator is paired
    # together, and then the second item in each passed iterator are paired
    # together etc. If the passed iterators have different lengths, the
    # iterator with the least items decides the length of the new iterator.
    # see here: https://www.w3schools.com/python/ref_func_zip.asp

    # because the groupby function can only be called on a DataFrame object,
    # and here we have a numpy array, it is easier to loop through a list
    # consisting of tuples that contain all the information we need to create
    # the plots
    for label,marker,color in zip(
        ["setosa", "versicolor","virginica"],('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x= X_lda_sklearn[:,0][Y == label],
                    y= X_lda_sklearn[:,1][Y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label)

    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title("Linear Discriminate Analysis")

    # hide axis ticks, perhaps a bit excessive, but oh well
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines, as they can be distracting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.savefig(plotDir + "LDAScatterPlot.png")
    plt.close()

    print(r"The LDA scatter plot for the Iris dataset has been created and saved to the 'plots\dimensionalityReduction' directory.")



def createPCAScatterPlot():

    # the code for this is adapted from here:
    # # https://sebastianraschka.com/Articles/2014_python_lda.html

    # we want to split the dataset into the variables proper
    # and the classes, i.e. the species. dataframe.iloc[]
    # can do this, which takes two parameters, the first slicing
    # the rows, the second the columns. Here we want all the rows,
    # and the first four columns. In the second case we only want the
    # fifth column
    X = df.iloc[:,0:4]
    Y = df.iloc[:,4]


    sklearn_pca = PCA(n_components=2)

    X_pca_sklearn = sklearn_pca.fit_transform(X, Y)

    ax = plt.subplot(111)
    for label,marker,color in zip(
        ["setosa", "versicolor","virginica"],('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x= X_pca_sklearn[:,0][Y == label],
                    y= X_pca_sklearn[:,1][Y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title("Principal Component Analysis")

    # hide axis ticks, perhaps a bit excessive, but oh well
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines, as they can be distracting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.savefig(plotDir + "PCAScatterPlot.png")
    plt.close()

    print(r"The PCA scatter plot for the Iris dataset has been created and saved to the 'plots\dimensionalityReduction' directory.")


def homemadeDimensionalityReduction():
    ax = plt.subplot(1,1,1)
    for label, group in (df.groupby("species")):
        # for the third parameter to the plot() function, we use '.'
        # to create a scatter plot, i.e. with dots rather than lines
        plt.plot(group["petal_width"]*300 + group["petal_length"]*75 + group["sepal_length"]*25 + group["sepal_width"]*25, group["sepal_length"], '.', label=label)

    # we add a legend to distinguish between the species, as they
    # are included on the same plot. We do not need to include the
    # label parameter, as we have already defined the labels in the
    # plot() function itself.
    # although the legends do not interfere with the points in the plots,
    # it is still safer to make them transparant.
    plt.legend(framealpha=0.5)
    plt.xlabel("Petal Width * 300 + Petal Length * 75 + Sepal Length * 25 + Sepal Width * 25")
    plt.ylabel("Sepal Length (cm)")
    plt.title("A Homemade Discriminant")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "homemadeDimensionalityReduction.png")
    # close the plot so that the next plot is not
    # superimposed on top of this one
    plt.close()

    print(r"A homemade dimensionality reduction for the Iris dataset has been created and saved to the 'plots\dimensionalityReduction' directory.")

if __name__ == '__main__':
    createLDAScatterPlot()
    createPCAScatterPlot()
    homemadeDimensionalityReduction()





# for eigenvectors: https://medium.com/fintechexplained/what-are-eigenvalues-and-eigenvectors-a-must-know-concept-for-machine-learning-80d0fd330e47

# for pca https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html

# for feature scaling https://benalexkeen.com/feature-scaling-with-scikit-learn/

# calculate components https://chrisalbon.com/machine_learning/feature_engineering/select_best_number_of_components_in_lda/


