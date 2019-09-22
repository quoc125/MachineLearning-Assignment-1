#Quoc Le
#3455489
#Use to find the k nearest neighbor 

import argparse
import numpy as np

class Iris:
  def __init__(self, sepal_length, sepal_width,
              petal_length, petal_width, type):
      self.sepal_length = sepal_length
      self.sepal_width = sepal_width
      self.petal_length = petal_length
      self.petal_width = petal_width
      self.type = type

#Return the cosine difference between two vector
def cosDif(v1,v2):
    numerator = np.dot(v1,v2)
    denominator = np.absolute(v1) * np.absolute(v1)
    return 1-np.arccos(numerator/denominator)

#Return the euclidean difference between two vector
def Euclidean(v1, v2):
    sum = 0
    for x, y in v1, v2:
        sum  = sum + np.square(y-x)
    return np.sqrt(sum)

def findNearNeighbor(training, test, k, method):

    return false

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
    testSet = []
    trainingSet = []
    size = int(len(List)/5)

    while (i < len(List)):
        trainingSet = (List[0:i])
        trainingSet.extend(List[i+size: len(List)])
        testSet = List[i : i+size]

        i = i + size
