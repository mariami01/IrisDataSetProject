import matplotlib.pyplot as plt

from createDataFrame import df


def createScatterPlots():

    cols=pltcolor(df["species"])

    # species

    plt.plot(df["species"], df["sepal_length"], ".")
    plt.xlabel("Species")
    plt.ylabel("Sepal Length (cm)")
    plt.title("Species vs. Sepal Length")
    plt.savefig("plots/scatterPlots/speciesSepalLength.png")
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


    # Sepal length

    plt.scatter(df["sepal_length"], df["sepal_width"], s=50, c=cols)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Sepal Length vs. Sepal Width")
    plt.savefig("plots/scatterPlots/sepalLengthSepalWidth.png")
    plt.close()

    plt.scatter(df["sepal_length"], df["petal_length"], s=50, c=cols)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title("Sepal Length vs. Petal Length")
    plt.savefig("plots/scatterPlots/sepalLengthPetalLength.png")
    plt.close()

    plt.scatter(df["sepal_length"], df["petal_width"], s=50, c=cols)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Sepal Length vs. Petal Width")
    plt.savefig("plots/scatterPlots/sepalLengthPetalWidth.png")
    plt.close()

    # Sepal width

    plt.scatter(df["sepal_width"], df["petal_length"], s=50, c=cols)
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title("Sepal Width vs. Sepal Length")
    plt.savefig("plots/scatterPlots/sepalWidthPetalLength.png")
    plt.close()

    plt.scatter(df["sepal_width"], df["petal_width"], s=50, c=cols)
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Sepal Width vs. Petal Width")
    plt.savefig("plots/scatterPlots/sepalWidthPetalWidth.png")
    plt.close()

    # Petal length

    plt.scatter(df["petal_length"], df["petal_width"], s=50, c=cols)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Petal Length vs. Petal Width")
    plt.savefig("plots/scatterPlots/petalLengthPetalWidth.png")
    plt.close()

# https://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l=='setosa':
            cols.append('red')
        elif l=='versicolor':
            cols.append('blue')
        else:
            cols.append('green')
    return cols
# Create the colors list using the function above

