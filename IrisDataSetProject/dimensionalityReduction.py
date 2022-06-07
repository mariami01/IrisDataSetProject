# for performing matrix operations
import numpy as np
import pandas as pd
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
# to test our models we will need to split our dataset in training
# and testing sets; we import train_test_split for this purpose
from sklearn.model_selection import train_test_split
# we also want to display the classification report
# and confusion matrix based on our tests
from sklearn.metrics import classification_report, confusion_matrix

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
    # we set the size of the test set as 0.2 of the dataset,
    # which is a generally well-performing proportion, and
    # we don't need to set the random seed parameter. See here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    Y_predictions = sklearn_lda.fit(X_train, Y_train).predict(X_test)
    # we print the confusion matrix
    print("Here is the confusion matrix for sklearn's LDA of the Iris dataset:")
    print(confusion_matrix(Y_test, Y_predictions))
    print()
    # followed by the classification report
    print("Here is the Classification Report for sklearn's LDA of the Iris dataset:")
    print(classification_report(Y_test, Y_predictions))

    # we now just plot these values as normal, using a scatter plot
    # so that the individual data points are distinguishable
    ax = plt.subplot(111)
    for label in ["setosa", "versicolor","virginica"]:

        plt.plot(X_lda_sklearn[:,0][Y == label],
                    X_lda_sklearn[:,1][Y == label],
                    '.',
                    label=label)

    plt.xlabel('Linear Discriminate 1')
    plt.ylabel('Linear Discriminate 2')

    plt.legend()
    plt.title("Linear Discriminate Analysis")

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
    # https://sebastianraschka.com/Articles/2014_python_lda.html

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
    for label in ["setosa", "versicolor","virginica"]:

        plt.plot(X_pca_sklearn[:,0][Y == label],
                    X_pca_sklearn[:,1][Y == label],
                    '.',
                    label=label)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.legend()
    plt.title("Principal Component Analysis")

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



def createManualLDA():
    # I have used many sources for achieving this, the main
    # one being here: https://sebastianraschka.com/Articles/2014_python_lda.html
    # to perform LDA, we need to calculate the within-class
    # and between-class scatter matricies, and then create
    # another matrix from these that we will decompose into
    # it eigenvalues, and then we will multiply the dataset
    # by the eigenvectors with the highest corresponding
    # eigenvalues

    # the formula for creating the within-class scatter matrix is:
    # the sum of (x−mi) (x−mi)Transposed for each row in each class,
    # where x is the row transposed and mi is an array holding the
    # means for each column for the current class.

    # As the within-class scatter matrix is calculated as a sum, we
    # initialize it as a matrix of zeros with dimension d*d, where d
    # is the number of features (columns) in the dataset
    withinClassScatterMatrix = np.zeros((4, 4))
    # create a 2d array containing arrays for the means of each class
    speciesMeans = []
    for species in ['setosa', 'versicolor', 'virginica']:
        speciesMeans.append(np.mean(df[df['species'] == species]).tolist())
    # we can use the zip function to iterate through the names of the classes
    # and their column means at the same time
    for species, mean in zip(['setosa', 'versicolor', 'virginica'], speciesMeans):
        # initialize the scatter matrix for the current class
        classScatterMatrix = np.zeros((4,4))
        # iterate through the rows of the current class
        for row in range(len(df[df['species'] == species])):
            row = df[df['species'] == species].iloc[row]
            # we now need to transpose the row into a d*1 matrix,
            # where d is the number of features in the dataset.
            row = np.array([row[:4]]).reshape(4,1)
            # same for the class means
            mean = np.array(mean).reshape(4,1)
            # now we just plug everything into the formula to create the
            # current classes scatter matrix
            classScatterMatrix = classScatterMatrix + (row-mean).dot((row-mean).T)
        # once the current classes scatter matrix is created, we add it to the
        # within-class scatter matrix
        withinClassScatterMatrix = withinClassScatterMatrix + classScatterMatrix
        # convert the numpy dtype of the matrix to float. We must do this
        # as the dtype of the df species are object, which will be problematic
        # later on when we are performing operations on the within-class scatter matrix.
    withinClassScatterMatrix = withinClassScatterMatrix.astype('float')

    # the formula for creating the between-class scatter matrix is:
    # the sum of each classes solution for Ni(mi−m) (mi−m)Transposed, where m is the overall mean,
    # mi and Ni are the means and sizes of the respective classes, and all three
    # are d*1 matrices, where d is the number of features in the dataset.
    # As the between class scatter matrix is calculated as a sum, we
    # initialize it as a matrix of zeros with dimension d*d, where d
    # is the number of features (columns) in the dataset
    betweenClassScatterMatrix = np.zeros((4, 4))
    # we need the means of all columns for this.
    # dataframe.mean() returns a series object, we want
    # to change this to a list with the series.tolist() method.
    means = df.mean().tolist()
    for i, mean in enumerate(speciesMeans):
        # we now need to transpose the 1*4 matrices of overall means
        # and means per species into 4*1 matricies.
        speciesMeans = np.array([mean]).reshape(4,1)
        overallMeans = np.array([means]).reshape(4,1)
        # and now we can simply plug everything into the formula
        betweenClassScatterMatrix += 50 * (speciesMeans - overallMeans).dot((speciesMeans - overallMeans).T)
    # now we want the product of the inverse of the covariance matrix and
    # the between class scatter matrix, and we will call this the
    # matrixToDecompose
    matrixToDecompose = np.linalg.inv(withinClassScatterMatrix).dot(betweenClassScatterMatrix)
    # we now need to decompose this matrix into its eigenvalues and
    # associated eigenvectors. This is a straightforward process
    # in that we can two equations to work with, and in both equations
    # there is only one unknown, namely, the eignvalues and eigenvectors
    # respectively. The equation to calculate the eigenvalues is:
    # determinant of (matrixToDecompose - eigenvalue(identity Matrix)) = 0
    # For calculating the eigenvectors we then use:
    # (matrixToDecompose - eigenvalue(identity Matrix))eigenvector = 0
    # However, there are two aspects here that are prohibitive,
    # namely the floating point values in the matrix will make the calculations
    # messy, and the first equation will result in polynomial equation of degree four.
    # For these reasons I am not going to spend time going through the calculations
    # step by step, but will simply let numpy do the work for me.
    eigenValues, eigenVectors = np.linalg.eig(matrixToDecompose)

    # What we can do at least is to plug the results eigenvalues and eigenvectors
    # back into the second of the two equations above, to assure ourselves
    # that nothing has gone wrong. We want to make sure that:
    # (matrixToDecompose - eigenvalue(identity Matrix))eigenvector = 0
    # we can use numpy's assert_array_almost_equal method for this
    # saw this here: https://sebastianraschka.com/Articles/2014_python_lda.html
    for i, e in enumerate(eigenValues):
        np.testing.assert_array_almost_equal(matrixToDecompose @ eigenVectors[:,i].reshape(4,1),
                                         e * eigenVectors[:,i].reshape(4,1),
                                         decimal=6, err_msg='', verbose=True)

    # now that we have the eigenvalues and eigenvectors, we want to rank the
    # eigenvalues and determine their associated 'explained variance', which
    # expresses (on a scale of 0-1) the model's ability to represent the data,
    # or more specifically the model's ability to explain the variance in the
    # variables in the dataset. Note that explained variance is referred
    # to as η squared (η**2), and its value is equal to the coefficient
    # of determination tha is used in regression analysis, R squared.
    # see here: https://www.statisticshowto.com/explained-variance-variation/
    # In understanding explained variance it is helpful to understand
    # what the eigenvalues actually mean here. The eigenvalues
    # represent the amount of variance in the variables in the dataset explained
    # by their corresponding eigenvectors. See here:
    # https://www.thejuliagroup.com/blog/factor-analysis-and-eigenvalues/
    # If you add up all of the eigenvalues, you get the total variance in the
    # dataset, and as such if you divide each eigenvalue by that total, one
    # arrives at the explained variance.
    # we don't really want to sort the eigenvalues here, because we want
    # their indexes to match the eigenvectors, so we just display them as is
    for i, j in enumerate(eigenValues):
        # perhaps due to Python's floating point imprecision, the eigenvalues
        # are actually complex, so we only want their real aspects
        print(f"Eigenvalue {i+1}'s explained variance: {j.real / sum(eigenValues.real)}")
    # In reality the first eigenvalue is sufficiently that we could capture
    # most of the data's variance with it alone, but we want to create a
    # scatter plot so we will include the next best eigenvector as a second
    # dimension. We now want to create the matrix that we will transform our
    # dataset with. This will be created from the eigenvectors corresponding
    # to the selected eigenvalues, such that the first column is composed of
    # the first eigenvectos and the second the second. To do this we first
    # transpose the eigenvectors to a 4*1 shape, and then pass then both to
    # numpy's hstack function, explained here: https://www.geeksforgeeks.org/numpy-hstack-in-python/
    # this function is capable of taking in arrays of arrays as arguments,
    # and adding the first arrays in each array together, and so on.
    transformativeMatrix = np.hstack((eigenVectors[:, 0].reshape(4,1).real, eigenVectors[:, 1].reshape(4,1).real))
    # now we simply multiply the dataset by this transformative Matrix
    # and plot to our heart's content
    # when multiply matrices the number of columns in the first
    # matrix must be equal to the number of rows in the second,
    # so I have removed the species column, to be added in after
    LDAspace = df.iloc[:, :4] @ transformativeMatrix
    # adding species column back in, code taken from here:
    # https://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas
    LDAspace['species'] = pd.Series(np.array(50 * ['setosa'] + 50 * ['versicollor'] + 50 * ['virginica']), index=df.index)
    # and now we just plot as normal
    ax = plt.subplot(111)
    for label in ["setosa", "versicolor", "virginica"]:

        plt.plot(LDAspace.iloc[:, 0][df['species'] == label],
                    LDAspace.iloc[:, 1][df['species'] == label],
                    '.',
                    label=label)

    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')

    plt.legend()
    plt.title("Manual Linear Discriminate Analysis")

    # remove axis spines, as they can be distracting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.savefig(plotDir + "manualLDA.png")
    plt.close()

    print(r"A manual linear discriminate analysis plot for the Iris dataset has been created and saved to the 'plots\dimensionalityReduction' directory.")

if __name__ == '__main__':
    createLDAScatterPlot()
    createPCAScatterPlot()
    homemadeDimensionalityReduction()
    createManualLDA()





# for eigenvectors: https://medium.com/fintechexplained/what-are-eigenvalues-and-eigenvectors-a-must-know-concept-for-machine-learning-80d0fd330e47

# for pca https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html

# for feature scaling https://benalexkeen.com/feature-scaling-with-scikit-learn/

# calculate components https://chrisalbon.com/machine_learning/feature_engineering/select_best_number_of_components_in_lda/
