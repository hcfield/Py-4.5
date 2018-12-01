    import pandas as pd
    import random as rnd
    import Node
    import math
    
    def classes(column):#gives the amount of times a label comes up in the dataset
        results = {}
        for row in column:
            if row not in results: results[row] = 0 
            results[row] += 1 
    
        return results #gives class and how many times it was there
    
    
    def Entropy(dataFile, column):#gets entropy
        ent = 0
        results = classes(dataFile[column]) # how many classes there are
    
        for row in results.values():
            p = float(row) / len(dataFile[column]) # probablity calculation
            ent -= p * math.log2(p) # entropy calculation
    
        return ent #gives entropy
    
    
    def thresholds(dataFile, column):#finds threshold points for a column
        
        sort= sorted[[column, 'type']] #column witht the header 'type'
        thresh = []
        p = sort[0] # first target
        index = sorted.index; # get the indexes of each  element in the sorted matrix
        counter = 0
        for row in sort:
            if row[1] != p: 
                thresh.append([index[counter], sorted[counter][0]])#separates data when class canges
            counter + 1
            p = row[1]
    
        return thresh #says where separations should occur
    
    
    # This divides the data into subsets depending on whether the data is above or below the threshold point
    def subData(dataSubs, column, thresh):
        belowThresh = []
        aboveThresh = []
        
        for i in range(len(thresh)):#threshold points - separates data into two parts at each point
            dataBelow = dataSubs[dataSubs[column] <= dataSubs[column][thresh[i][0]]]  # everything below the splitpoint
            dataAbove = dataSubs[dataSubs[column] > dataSubs[column][thresh[i][0]]]  # everything above it
            
            belowThresh.append(dataBelow)#adds to the above and below TH lists
            aboveThresh.append(dataAbove)
    
        return belowThresh, aboveThresh #returns the lists containing data above and below TH's
    
    
    def infoGain(dataF, column):#information gain
        
        aboveOccs = []
        belowOccs = []
        aboveEnt = []
        belowEnt = []
        totalOccs = []
        infoGain = []
        
        thresholds = findSplitPoints(dataF, column)  # thresholds for chosen column
        belowThresh, aboveThresh = subData(dataF, column, thresholds) #separates data by way of the thresholds  
           
        entropy = Entropy(dataF, 'type')  # get target entropy for the dataset
        
        for set in aboveThresh:
            aboveEnt.append(Entropy(set, 'type'))
            aboveOccs.append(len(set)) #entropy for data above threshold
        for set in belowThresh:
            belowEnt.append(Entropy(set, 'type'))
            belowOccs.append(len(set)) #entropy for data below threshold
    
        
        
        for i in range(len(belowOccs)):
            totalOccs.append(belowOccs[i] + aboveOccs[i])
            A = (aboveOccs[i] / float(totalOccs[i]))#info gain for above the threshold
            B = (belowOccs[i] / float(totalOccs[i]))#info gain for below the threshold
            infoGain.append(entropy - ((belowEnt[i] * B) + (aboveEnt[i] * A)))
    
        
        optimumGain = i = counter = 0 #optimum gain = best IG for the chosen column
        for gain in infoGain:
            if optimumGain < gain:
                optimumGain = gain
                counter = i # holds where optimum gain occurs
            i += 1
    
        #belowThresh=subset below with best gain/ same for others
        return optimumGain, belowThresh[counter], aboveThresh[counter], thresholds[counter]
    
    
    def designtree(dataTree):#designs tree
        
        best = {}
        columns = []#empty list
        i = 0
        highestGain = -1
    
        for column in dataTree:  
            if column != 'type':
                try:
                    infoG, data0, data1, sep = infoGain(dataTree, column)  #retrieves the IG for each column in the tree datad
                    
                    #creates a list so these attributes can be used in a tree
                    columns.append({"infoG": infoG, "left set": data0, "right set": data1, 'column': i, 'separate': sep,
                                    'colName': column})  
    
                except IndexError:#used for when an element that cant be viewed arrives, uses data for leaf node
                    columns.append({"infoG": 0, "left set": [], "right set": [], 'column': column, })
            i += 1  
    
        
        for values in range(len(columns)):
    
            if columns[values]['infoG'] > highestGain:
                best = columns[values]
                highestGain = columns[values]['infoG']#takes best info gain
    
        
        left = best['left set'] #left set data
        right = best['right set']#right set data
        
        
        if len(best['left set']) != 0 and len(best['right set']) != 0:#makes sure there is still data
            return (Node.node(col=best['column'], colName=best['colName'], value=best['separate'][1], results=None,
                                  rCh=designtree(right), lCh=designtree(left)))
    
        else: #arrived at end point so leaf node is created
            label = list(classes(dataTree['type']).keys()); 
            return (Node.node(results=label[0]))
        
        
    
    def classifying(row, tree):#finds leaf node that row will classify as
        
        if tree.results != None:
            return tree.results#shows when the program arrives at a leaf node
    
        else:
            
            tvalue = row[tree.col]
            sets = None
            if isinstance(tvalue, float):
               
                if tvalue >= tree.value:#if tree value is less than attribute value it goes down right side
                    sets = tree.rCh
                
                else:#otherwise it goes down the left side
                    sets = tree.lCh
            
            return classifying(row, sets)
        
    
    def Tree(dataTest, labels, tree):#tests tree and the classifier
        values = []
        indexes = labels.index
        correct = incorrect = 0 #assigns 0 to incorrect and correct
        
        for index, row in dataTest.iterrows():
           values.append([index, classifying(row, tree)])#stores the result for each index
    
        for j in range(len(values)):
            if values[j][0] == indexes[j] and values[j][1] == labels[indexes[j]]:
                correct += 1 #adds to the amount of correct classification
            else:
                incorrect += 1 #adds to the amount of incorrect classifications
    
        return correct, incorrect, (1 - (incorrect / (incorrect + correct)) )*100 #returns a % which is the accuracy
    
    def main():
    
        dataSet = pd.read_csv('C:\\Users\\User\\Documents\\Engineering\\4th year\\ML DM\\owls15.csv')#owls dataset, other csv files could be used
    
        acc = [];
        for i in range(10):#number of times the loop will be completed creating new datasets each time to achieve an avg accuracy
            
            
            ratio=1/3 #this ratio is used to divide the data into a train and test set
            def trainTestData(dataFile, ratio):#used to create the train and test data set
                length = range(len(dataFile)) 
                test = rnd.sample(length, int(ratio * len(dataFile))) # get a random sample of indices from this array and by multiplying by % split we want
                train = list(set(length) ^ set(test)) # take away the test indexes from the the train indexes using the intersection between the sets total and train
            
                testFile = dataFile.loc[test]
                trainFile = dataFile.loc[train]
            
                return testFile, trainFile
            
            testData, trainData  = trainTestData(dataSet, ratio); 
            tree = designtree(trainData) # creates the tree
            types = testData['type'] # retrieves the data from the 'types' column in the testing data
    
            incorrect, correct, accuracy = Tree(testData, types, tree) 
            acc.append(accuracy)
    
            
            print(str(i + 1) + "\n")#prints what loop the program is on
            print("Correctly Classified: " + str(correct) + " / " + str(correct+incorrect))#prints how many correct classifications there were
            print("Accuracy: " + str(accuracy))#pritns the accuracy
            
        sum = 0
        for res in range(len(acc)):
            sum += acc[res]
        average = sum/10
    
        print("Avg Accuracy = " )
        print(average)#gives the average accuracy after all 10 folds have been completed
    
    
    if __name__ == '__main__':
        main()#runs main function