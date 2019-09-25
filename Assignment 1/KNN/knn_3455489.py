#Quoc Le
#3455489
#Use to find the k nearest neighbor 

import argparse
import numpy as np

#class to store Iris infomation
class Iris:
  def __init__(self, sepal_length, sepal_width,
              petal_length, petal_width, type):
      self.sepal_length = float(sepal_length)
      self.sepal_width = float(sepal_width)
      self.petal_length = float(petal_length)
      self.petal_width = float(petal_width)
      self.type = type
      self.dataList = np.zeros(4, dtype=float)
      self.dataList[0] =  self.sepal_length
      self.dataList[1] =  self.sepal_width
      self.dataList[2] =  self.petal_length
      self.dataList[3] =  self.petal_width


#Return the cosine difference between two vector
def cosDif(v1,v2):
    numerator = np.dot(v1,v2)
    denominator = absolute(v1) * absolute(v2)
    return 1-numerator/denominator

#Give the absolute value of the vector
def absolute(v1):
    sum = 0
    for i in range(4):
        sum = sum + np.square(v1[i])
    return np.sqrt(sum)

#Return the euclidean difference between two vector
def Euclidean(v1, v2):
    sum = 0
    for i in range(4):
        sum  = sum + np.square(v1[i]-v2[i])
    return np.sqrt(sum)

def Predict(voter):
    total = 0
    convert = {"Iris-setosa":0,  "Iris-versicolor":1, "Iris-virginica":2}
    matrix = [[0,0,0],[0,0,0],[0,0,0]]
    for input in voter:
        count = {}
        for i in range(len(voter[input])):
            if voter[input][i][1] not in count:
                count[voter[input][i][1]] = 0
            count[voter[input][i][1]] =  count[voter[input][i][1]] +1
        winner = None
        TopCount = 0
        for type in count:
            if TopCount < count[type]:
                winner = type
                TopCount = count[type]
        matrix[convert[winner]][convert[input.type]] = matrix[convert[winner]][convert[input.type]] + 1
        if winner == input.type:
            total = total +1
    accuracy = float(total)/ len(voter)    
    return accuracy

def findNearNeighbor(training, test, k, method):
    voter = {}
    for y in test:
        voter[y] = []
        for i in range(k):
            voter[y].append( [99999, None])
        for val in training:
            distance = 0
          
            if method == "Euclidean":
                distance = distance + Euclidean(val.dataList, y.dataList)
            elif method == "Cosine":
                distance = distance + cosDif(val.dataList, y.dataList)

            for i in range(k):
                if float(voter[y][i][0]) > distance:
                    voter[y][i][0] = distance
                    voter[y][i][1] = val.type
                    break;       
    return voter

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--k")
    ap.add_argument("-m", "--method")
    args = vars(ap.parse_args())

    List = []
    with open('iris.data') as fp:
       line = fp.readline()
       while line:
           parameter = line.strip().split(',')
           if (len(parameter) == 5):
                List.append(Iris(parameter[0],parameter[1],parameter[2], parameter[3], parameter[4]))
           line = fp.readline()
    
    i = 0
    j = 0
    testSet = []
    trainingSet = []
    crossValidation = []
    size = int(len(List)/5)
    accuracy = 0
    while (i < len(List)):
        trainingSet = (List[0:i])
        trainingSet.extend(List[i+size: len(List)])
        testSet = List[i : i+size]
        voter = findNearNeighbor(trainingSet,testSet,int(args["k"]), args["method"])
        tempAccuracy= Predict(voter)
        accuracy = accuracy + tempAccuracy/5

        j = j +1
        i = i + size
    print(args["k"] + " nearest neighbors using " +args["method"] + " difference with an accuracy of " +str(accuracy * 100) + "%")