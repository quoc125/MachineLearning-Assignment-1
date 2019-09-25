#Quoc Le
#3455489
#Naive Bayes
import numpy as np

class Iris:
    def __init__(self, sepal_length, sepal_width,
              petal_length, petal_width, type):
      self.sepal_length = sepal_length
      self.sepal_width = sepal_width
      self.petal_length = petal_length
      self.petal_width = petal_width
      self.type = type
      self.dataList = [sepal_length,sepal_width, petal_length, petal_width]

#Split training set for each class
def splitData(data):
    newData = {}
    for x in data:
        if (x.type not in newData):
            newData[x.type] = []       
        newData[x.type].append(x.dataList)
    return newData

#Calculate the mean  and return count and mean
def calMean(data):
    means = {}
    for y in data:
        if y not in means:
            means[y] = [0,0,0,0]
        for i in range(len(data[y])):
            for j in range(4):
                means[y][j] = means[y][j] +float(data[y][i][j])/len(data[y])    
    return means

#Calcuate the standard deviation 
def calStdDev(mean,data):
    stdDev = {}
    for y in data:
        if y not in stdDev:
            stdDev[y] = [0,0,0,0]
        for i in range(len(data[y])):
            for j in range(4):
               stdDev[y][j] = stdDev[y][j] + (np.square(mean[y][j] - float(data[y][i][j])))/(len(data[y])-1)
        for i in range(4):
            stdDev[y][i] = np.sqrt(stdDev[y][i])
    return stdDev

#Calculate the probablity 
def calProb(x, mean, stdDev, training):
    classProp = {}
    for y in mean:
        if y not in classProp:
            classProp[y] = [0,0,0,0]
        for i in range(4):          
            classProp[y][i] = np.exp(-1 * (np.square(float(x.dataList[i]) - mean[y][i]))/(2*np.square(stdDev[y][i])))/(np.sqrt(2 *np.pi)*stdDev[y][i])
    prop = {}
    for y in classProp:
        if y not in prop:
            prop[y] = len(training[y])/130 
        for i in range(4):
            prop[y] = prop[y] * classProp[y][i]
    normalize = 0
    for y in prop:
        normalize = normalize+ prop[y]
    for y in prop:
        prop[y] = prop[y]/normalize
    return prop

#Make prediction
def Predict(prop):
    bestLabel = None
    bestValue = 0
    for y in prop:
        if bestValue < prop[y]:
            bestLabel = y
            bestValue = prop[y]
    return bestLabel

if __name__ == "__main__":
    List = []
    with open('iris.data') as fp:
       line = fp.readline()
       while line:
           parameter = line.strip().split(',')        
           if (len(parameter) == 5):
                List.append(Iris(parameter[0],parameter[1],parameter[2], parameter[3], parameter[4]))
           line = fp.readline()
    

    i = 0
    size = int(len(List)/5)

    testSet = []
    trainingSet = []
    convert = {"Iris-setosa":0,  "Iris-versicolor":1, "Iris-virginica":2}
    accuracy = 0

    while i < len(List):
        trainingSet = (List[0:i])
        trainingSet.extend(List[i+size: len(List)])
        testSet = List[i : i+size]
        matrix = [[0,0,0],[0,0,0],[0,0,0]]
        count = 0

        splitTrainingData = splitData(trainingSet)
        mean = calMean(splitTrainingData)
        stdDev = calStdDev(mean,splitTrainingData)
        for data in testSet:        
            prop = calProb(data, mean, stdDev, splitTrainingData)
            prediction = Predict(prop)
            if prediction == data.type:
                count  = count +1
            matrix[convert[prediction]][convert[data.type]] = matrix[convert[prediction]][convert[data.type]] +1
        accuracy = accuracy +(count/len(testSet))/5
        i = i + size

    print("Naive Bayes had an accuracy of " + str(accuracy*100) + "%")

