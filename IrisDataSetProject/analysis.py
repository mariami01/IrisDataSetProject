# While the other files in the repository can be run
# on their own, by running this file one can run them
# all from a central location.

from verifyIntegrityOfDataset import verifyIntegrityOfDataset
from createSummaryStatistics import createSummaryStatistics
from createHistograms import createHistograms
from createBoxPlots import createBoxPlots
from createScatterPlots import createScatterPlots
from dimensionalityReduction import createLDAScatterPlot
from dimensionalityReduction import createPCAScatterPlot
from dimensionalityReduction import homemadeDimensionalityReduction
from dimensionalityReduction import createManualLDA


def main():
    verifyIntegrityOfDataset()
    createSummaryStatistics()
    createHistograms()
    createBoxPlots()
    createScatterPlots()
    createLDAScatterPlot()
    createPCAScatterPlot()
    homemadeDimensionalityReduction()
    createManualLDA()

if __name__ == '__main__':
        main()
