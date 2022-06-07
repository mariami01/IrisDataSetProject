# for performing matrix operations
import numpy as np
# for saving plots etc.
import matplotlib.pyplot as plt
# get the dataframe from the createDataFrame file
# we will use seaborn to display a heat map of the correlation matrix
import seaborn as sns
from createDataFrame import df
# we will use the os module to create the
# the directory to store the files to
import os

def linearRegression():

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
    plotDir = os.path.join(scriptDir, 'plots/linearRegression/')
    # we then check if the directory we want to create already
    # exists, and only if it doesn't exist do we invoke the os
    # module's makedirs() method to create it.
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)
    global df
    # dataframe.mean() returns a series object, we want
    # to change this to a list with the series.tolist() method.
    means = df.mean().tolist()
    # we now remove the species column from the dataframe
    variablesMatrix = df.iloc[:, :4]
    # to create the 'means matrix' we create a matrix
    # the same size as the variable matrix but filled with ones
    meansMatrix = np.ones((150, 4))
    # we then simply multiply this matrix of ones by the list
    # of means.
    meansMatrix *= means
    # subtracting the means matrix from the variables matrix
    # gives us the zero-centred matrix
    zeroCentredMatrix = variablesMatrix - meansMatrix
    # now multiply the variablesMatrix but its transposition
    transpositionMatrixByVariablesMatrix = variablesMatrix.transpose() @ variablesMatrix
    # to create the 'means product matrix' we multiply the transpose
    # of the means by the untransposed matrix of means - note that
    # this is element-wise rather than matrix multiplication
    meansProductMatrix = np.array([means]).T * np.array([means])
    # we now multiply this matrix by the scalar, number of instances
    nMeansProductMatrix = 150 * meansProductMatrix
    # to get the scatter matrix we subtract the above matrix from the
    # transpositionMatrixByVariablesMatrix
    scatterMatrix = transpositionMatrixByVariablesMatrix - nMeansProductMatrix
    # to get the covariance matrix we divide the scatter matrix by the scalar,
    # number of instances
    covarianceMatrix = (1 / 150) * scatterMatrix
    # we now want to check that the difference between the values in
    # our covariance matrix and the matrix as calculated by Pandas itself
    # is not significant. We can do this by
    # calculating the mean difference between the covariance matrix
    # values as calculated here and as calculated by Pandas cov()
    # method. We do this by calling Dataframe.mean on the dataframe,
    # which will return a series object of the mean for each column,
    # and then calling series.mean will return the mean of these means.
    # we want to make sure the difference is less than 0.01, so we call
    # abs() on the number.
    assert abs(((covarianceMatrix - df.cov()).mean()).mean()) < 0.01
    # to get the correlation matrix we first need to calculate the
    # standard deviation of each feature.
    stds = df.std().tolist()

    correlationMatrix = covarianceMatrix.copy()
    # we now divide the covariances by the produce of the variables
    # standard deviation. This can be done manually with nested for loops.
    for i in range(len(covarianceMatrix)):
        for j in range(len(stds)):
            x = covarianceMatrix.iloc[i][j]
            y =  stds[i]
            correlationMatrix.iloc[i][j] /= (stds[i] * stds[j])
    # we now check that this matches with Pandas own calculation
    assert abs(((correlationMatrix - df.corr()).mean()).mean()) < 0.01

    # of course, where performing actual analysis, one would just call
    # the dataframe.corr() method to create the correlation matrix,
    # and then use seaborn's heatmap method to display it.
    # we pass the additional parameter: annot=True so that the actual
    # correlation scores are shown for each square, where the default
    # is to two decimal places; vmin, vmax, center define the scale
    # of the heatmap, the default is to have the correlations themselves
    # determine the min and max, but I think it is clear if we make the
    # min and max mirror the actual limits of the magnitude of
    # the correlation coefficients. cmap='coolwarm' to change the color
    # scheme (I like coolwarm because the high correlations are
    # clearly visible as either or blue); square=True to make the
    # sections square rather than the default rectangle;
    heatMap = sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, center= 0, cmap='coolwarm', square=True)
    # note that there is a bug in the matplotlib regression between 3.1.0 and 3.1.1
    # such that the bottom and top sections appear to be cut off in
    # seaborn heatmaps. This can be fixed by manually getting the y-
    # limits of the map and then resetting them with half a section
    # in height added, as shown here:
    # https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    bottom, top = heatMap.get_ylim()
    heatMap.set_ylim(bottom + 0.5, top - 0.5)
    # the ylabels are by default written vertically, so we want
    # to change this by setting their rotation to zero, as shown here:
    # https://likegeeks.com/seaborn-heatmap-tutorial/
    heatMap.set_yticklabels(heatMap.get_yticklabels(), rotation=0)
    # because the labels are quite lengthy, it is also clearer if we
    # rotate the xlabels slightly
    heatMap.set_xticklabels(heatMap.get_xticklabels(), rotation=10)
    plt.title("Correlation Matrix for All Species")
    plt.savefig(plotDir + "correlationHeatMap.png")
    plt.close()

    # and this is so easy that one might as well do it for each species
    for label, group in df.groupby("species"):
        heatMap = sns.heatmap(group.corr(), annot=True, vmin=-1, vmax=1, center= 0, cmap='coolwarm', square=True)
        bottom, top = heatMap.get_ylim()
        heatMap.set_ylim(bottom + 0.5, top - 0.5)
        heatMap.set_yticklabels(heatMap.get_yticklabels(), rotation=0)
        heatMap.set_xticklabels(heatMap.get_xticklabels(), rotation=10)
        if label == 'setosa':
            plt.title("Correlation Matrix for Setosas")
            plt.savefig(plotDir + "correlationHeatMapSetosa.png")
            plt.close()
        elif label == 'versicolor':
            plt.title("Correlation Matrix for Versicolors")
            plt.savefig(plotDir + "correlationHeatMapVersicolor.png")
            plt.close()
        else:
            plt.title("Correlation Matrix for Virginicas")
            plt.savefig(plotDir + "correlationHeatMapVirginica.png")
            plt.close()





    # we can go a step further and calculate a 'determination' matrix
    # by squaring the correlation coefficients
    determinationMatrix = correlationMatrix.copy()
    for i in range(len(determinationMatrix)):
        for j in range(len(determinationMatrix)):
            determinationMatrix.iloc[i][j] *= determinationMatrix.iloc[i][j]


    # calculate the regression coefficient where petal length is the
    # explanatory variable and petal width is the dependent variable

    regressionCoefficient = covarianceMatrix.iloc[2][3] / covarianceMatrix.iloc[2][2]
    yIntercept = means[3] - regressionCoefficient * means[2]

    # because the slope of a straight line does not change, we only
    # need to use two two y values as predicted by the linear regression
    # equation to plot the line, namely, those predicted from the lowest
    # and highest values of x respectively
    predictedY = [yIntercept + regressionCoefficient * df['petal_length'].min(), yIntercept + regressionCoefficient * df['petal_length'].max()]

    ax = plt.subplot(1,1,1)
    for label, group in df.groupby("species"):
        plt.plot(group["petal_length"], group["petal_width"], '.', label=label)
    plt.plot([df['petal_length'].min(), df['petal_length'].max()], predictedY)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Linear Regression of Petal Length vs. Petal Width")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "linearRegressionPetalLengthPetalWidth.png")
    plt.close()

    residuals = [1 for x in df['petal_length']]
    for i in range(len(df['petal_length'])):
        residuals[i] = (yIntercept + regressionCoefficient * df['petal_length'][i]) - df['petal_width'][i]


    ax = plt.subplot(1,1,1)
    plt.plot(df["petal_length"], residuals, '.', label=label)
    plt.title("Residual Plot for Linear Regression of Petal Length vs. Petal Width")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(plotDir + "residualPlotPetalLengthPetalWidth.png")
    plt.close()










    ax = plt.subplot(2,2,1)
    plt.plot(df["petal_length"], residuals, '.', label=label)
    plt.ylabel("Residuals")
    plt.title("All species")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


    df1 = df[df['species'] == 'setosa']
    means = df1.mean().tolist()
    variablesMatrix = df1.iloc[:, :4]
    meansMatrix = np.ones((50, 4))
    meansMatrix *= means
    zeroCentredMatrix = variablesMatrix - meansMatrix
    transpositionMatrixByVariablesMatrix = variablesMatrix.transpose() @ variablesMatrix
    meansProductMatrix = np.array([means]).T * np.array([means])
    nMeansProductMatrix = 50 * meansProductMatrix
    scatterMatrix = transpositionMatrixByVariablesMatrix - nMeansProductMatrix
    covarianceMatrix = (1 / 50) * scatterMatrix
    #assert abs(((covarianceMatrix - df1.cov()).mean()).mean()) < 0.01
    stds = df.std().tolist()
    correlationMatrix = covarianceMatrix.copy()
    for i in range(len(covarianceMatrix)):
        for j in range(len(stds)):
            x = covarianceMatrix.iloc[i][j]
            y =  stds[i]
            correlationMatrix.iloc[i][j] /= (stds[i] * stds[j])
    #assert abs(((correlationMatrix - df1.corr()).mean()).mean()) < 0.01
    determinationMatrix = correlationMatrix.copy()
    for i in range(len(determinationMatrix)):
        for j in range(len(determinationMatrix)):
            determinationMatrix.iloc[i][j] *= determinationMatrix.iloc[i][j]
    regressionCoefficient = covarianceMatrix.iloc[2][3] / covarianceMatrix.iloc[2][2]
    yIntercept = means[3] - regressionCoefficient * means[2]
    predictedY = [yIntercept + regressionCoefficient * df1['petal_length'].min(), yIntercept + regressionCoefficient * df1['petal_length'].max()]

    residuals = [1 for x in df1['petal_length']]
    for i in range(len(df1['petal_length'])):
        residuals[i] = (yIntercept + regressionCoefficient * df1['petal_length'].iloc[i]) - df1['petal_width'].iloc[i]

    ax = plt.subplot(2,2,2)
    plt.plot(df1["petal_length"], residuals, '.', label=label)
    plt.title("Setosa")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)





    df1 = df[df['species'] == 'versicolor']
    means = df1.mean().tolist()
    variablesMatrix = df1.iloc[:, :4]
    meansMatrix = np.ones((50, 4))
    meansMatrix *= means
    zeroCentredMatrix = variablesMatrix - meansMatrix
    transpositionMatrixByVariablesMatrix = variablesMatrix.transpose() @ variablesMatrix
    meansProductMatrix = np.array([means]).T * np.array([means])
    nMeansProductMatrix = 50 * meansProductMatrix
    scatterMatrix = transpositionMatrixByVariablesMatrix - nMeansProductMatrix
    covarianceMatrix = (1 / 50) * scatterMatrix
    #assert abs(((covarianceMatrix - df1.cov()).mean()).mean()) < 0.01
    stds = df.std().tolist()
    correlationMatrix = covarianceMatrix.copy()
    for i in range(len(covarianceMatrix)):
        for j in range(len(stds)):
            x = covarianceMatrix.iloc[i][j]
            y =  stds[i]
            correlationMatrix.iloc[i][j] /= (stds[i] * stds[j])
    #assert abs(((correlationMatrix - df1.corr()).mean()).mean()) < 0.01
    determinationMatrix = correlationMatrix.copy()
    for i in range(len(determinationMatrix)):
        for j in range(len(determinationMatrix)):
            determinationMatrix.iloc[i][j] *= determinationMatrix.iloc[i][j]
    regressionCoefficient = covarianceMatrix.iloc[2][3] / covarianceMatrix.iloc[2][2]
    yIntercept = means[3] - regressionCoefficient * means[2]
    predictedY = [yIntercept + regressionCoefficient * df1['petal_length'].min(), yIntercept + regressionCoefficient * df1['petal_length'].max()]

    residuals = [1 for x in df1['petal_length']]
    for i in range(len(df1['petal_length'])):
        residuals[i] = (yIntercept + regressionCoefficient * df1['petal_length'].iloc[i]) - df1['petal_width'].iloc[i]

    ax = plt.subplot(2,2,3)
    plt.plot(df1["petal_length"], residuals, '.', label=label)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Residuals")
    plt.title("Versicolor")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)






    df1 = df[df['species'] == 'virginica']
    means = df1.mean().tolist()
    variablesMatrix = df1.iloc[:, :4]
    meansMatrix = np.ones((50, 4))
    meansMatrix *= means
    zeroCentredMatrix = variablesMatrix - meansMatrix
    transpositionMatrixByVariablesMatrix = variablesMatrix.transpose() @ variablesMatrix
    meansProductMatrix = np.array([means]).T * np.array([means])
    nMeansProductMatrix = 50 * meansProductMatrix
    scatterMatrix = transpositionMatrixByVariablesMatrix - nMeansProductMatrix
    covarianceMatrix = (1 / 50) * scatterMatrix
    #assert abs(((covarianceMatrix - df1.cov()).mean()).mean()) < 0.01
    stds = df.std().tolist()
    correlationMatrix = covarianceMatrix.copy()
    for i in range(len(covarianceMatrix)):
        for j in range(len(stds)):
            x = covarianceMatrix.iloc[i][j]
            y =  stds[i]
            correlationMatrix.iloc[i][j] /= (stds[i] * stds[j])
    #assert abs(((correlationMatrix - df.corr()).mean()).mean()) < 0.01
    determinationMatrix = correlationMatrix.copy()
    for i in range(len(determinationMatrix)):
        for j in range(len(determinationMatrix)):
            determinationMatrix.iloc[i][j] *= determinationMatrix.iloc[i][j]
    regressionCoefficient = covarianceMatrix.iloc[2][3] / covarianceMatrix.iloc[2][2]
    yIntercept = means[3] - regressionCoefficient * means[2]
    predictedY = [yIntercept + regressionCoefficient * df1['petal_length'].min(), yIntercept + regressionCoefficient * df1['petal_length'].max()]

    residuals = [1 for x in df1['petal_length']]
    for i in range(len(df1['petal_length'])):
        residuals[i] = (yIntercept + regressionCoefficient * df1['petal_length'].iloc[i]) - df1['petal_width'].iloc[i]

    ax = plt.subplot(2,2,4)
    plt.plot(df1["petal_length"], residuals, '.', label=label)
    plt.xlabel("Petal Length (cm)")
    plt.title("Virginica")
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    plt.savefig(plotDir + "residualplots.png")







if __name__ == '__main__':
    linearRegression()
