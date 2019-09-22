#Quoc Le
#3455489
#Naive Bayes

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
    testSet = []
    trainingSet = []
    size = int(len(List)/5)
