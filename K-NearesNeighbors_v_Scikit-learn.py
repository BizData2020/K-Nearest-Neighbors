# Supervised Machine Learning:
# Compare K-Nearest Neighbors function to Scikit-Learn's version
# for Cancer Prediction Modeler

import sys
import os
import numpy as np
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
import warnings
from math import sqrt
from collections import Counter
from sklearn import preprocessing, cross_validation, neighbors
from matplotlib import style


"""
Data Source:

http://archive.ics.uci.edu/ml/datasets.html

http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

7. Attribute Information: (class attribute has been moved to last column)

   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
"""

############
# here's our hand-written K Nearest Neighbors function:
#
# confidence can come from the classifier
#
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value that is less than total voting groups')
    
# we have to compare the prediction point to 
# all the other data pts in the input data
#
    distances = []
    for group in data:
        for features in data[group]:
# 2 dimension version is hard coded, ng:
#            euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )

# N-dimension version:
#            euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))

# Here's a third even faster way to calc Euclidean Distance:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
        
    votes = [i[1] for i in sorted(distances)[:k]]
    
#    print("Counter(votes).most_common(1): ", Counter(votes).most_common(1))
    
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
#    print("vote_result, confidence: ", vote_result, confidence)

    return vote_result, confidence


################
# Setup the path to our data files

# c:\\ML_Data\\_UCI\\BreastCancer_orig\\breast-cancer-wisconsin_data.txt

cwd = os.getcwd()
#print(cwd)
os.chdir('c:\\ML_Data\\_UCI\\BreastCancer_orig\\')
cwd2 = os.getcwd()
#print(cwd2)

accuracies = []

# Loop through to compare accuracies:
#
for i in range(25):
    # read in our data file
    #
    df = pd.read_csv("breast-cancer-wisconsin_data.txt")

    # convert unknown data vals to huge outlier
    # we could also just drop data rows that have a '?' 
    #
    df.replace('?', -99999, inplace=True) 

    # need to drop the ID since it has no impact on 
    # benign vs malignant
    #
    df.drop(['id'], 1, inplace=True)

    #print("Head ints1: ", df.head(5))

    # our sample data is all ints, but to make
    #  the code re-useable, onvert the complete 
    # dataframe data to float

    full_data = df.astype(float).values.tolist()

    #print("full_data: ", full_data[:5])

    # shuffle our data before training
    #
    random.shuffle(full_data)

    #print("full_data:rnd ", full_data[:5])

    # now prep for training
    #
    test_size = 0.2     # train with 80% of the data, test with 20%
    train_set = {2:[], 4:[]}
    test_set  = {2:[], 4:[]}

    # our training data is the first 80% of full_data
    #
    train_data = full_data[:-int(test_size*len(full_data))]

    # our test data is the last 20% of full_data
    #
    test_data = full_data[-int(test_size*len(full_data)):]

    # Now we need to populate the dictionaries
    #
    # Note: train_set[i[-1]] is the last element: 2 benign, 4 malig.
    #
    for i in train_data:
        train_set[i[-1]].append(i[:-1]) # append all features except label

    #now do the same thing for the test data:
    #
    for i in test_data:
        test_set[i[-1]].append(i[:-1]) # append all features except label

    # we've now populated or train and test dictionaries
    # and we're ready to call our hand written K-NN function

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
    #        else:
    #            print("Confidence: ", confidence) # print confidence scores of incorrect votes
            total += 1

    print("Accuracy fr Func x25: ", correct / total)         
    accuracies.append(correct / total)
   
# print out our average accuracy after 25 runs:
#
print("Avg. Accuracy: fr Func ", sum(accuracies) / len(accuracies))

# for a run of 25 times, Avg Accuracy from our KNN funct is:
# Avg. Accuracy: fr Func  0.968345323741007
#
# Our KNN function produces similar results to:
# neighbors.KNeighborsClassifier(), but runs much slower
# since: neighbors.KNeighborsClassifier() is threaded
#
# With: neighbors.KNeighborsClassifier(), each set of 
# features is it's own snowflake/thread
#
# Also: neighbors.KNeighborsClassifier() is faster 
# as it has a radius param to ignore values outside
# the radius; also faster since we set n_jobs to -1
# to be multi-threaded
#

################ done with i in range(25) ####################


############## Start of testing using neighbors.KNeighborsClassifier()

accuracies = []

# Loop through to compare accuracies:
#
for i in range(25):
    
    # read in our data file

    df = pd.read_csv("breast-cancer-wisconsin_data.txt")

    # convert unknown data vals to huge outlier
    # we could also just drop data rows that have a '?' 
    #
    df.replace('?', -99999, inplace=True) 

    # need to drop the ID since it has no impact on 
    # benign vs malignant
    df.drop(['id'], 1, inplace=True)

    #print(df.head(5))


    # Now define X (features) and y (label(s))
    # Note: most of the following code is very 
    # similar to LinearRegression code

    X = np.array(df.drop(['class'],1))  # our features

    y = np.array(df['class'])           # our labels

    # Cross validate: with 20% var
    # this shuffles the data into training and testing chunks: 80% train, 20% test
    #
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # define our classifier:
    # pass: n_jobs=-1 for multi-threading
    # with local computer, not passing n_jobs=-1 
    # is actually faster
    clf = neighbors.KNeighborsClassifier() 

    clf.fit(X_train, y_train)

    # don't confuse accuracy with confidence
    #
    accuracy = clf.score(X_test, y_test)

    print("Accuracy fr KNN() x25: ", accuracy)

    accuracies.append(accuracy)


print("Avg. Accuracy: fr neighbors.KNeighborsClassifier() x25", sum(accuracies) / len(accuracies))

# for a run of 25 times, Avg Accuracy from KNeighborsClassifier() is:
# Avg. Accuracy: fr ScikitLearn x25 0.966571428571

# NOTE: Can use KNN() on non-linear data where regression() wouldn't work

############## End of testing using neighbors.KNeighborsClassifier()


# Let's see if we increase k above, would
# our accuracy go up? In testing a higher k
# didn't see to help much, so stay with: k=5


###################################### Previous Session ###############
#
style.use('fivethirtyeight')

# Next: we want to compare to ScikitLearn results
# to see which is more accurate
#


# Now define X (features) and y (label(s))
# Note: most of the following code is very 
# similar to LinearRegression code

X = np.array(df.drop(['class'],1))  # our features

y = np.array(df['class'])           # our labels

# Cross validate: with 20% var
# this shuffles the data into training and testing chunks: 80% train, 20% test
#
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# define our classifier:
#
clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

# don't confuse accuracy with confidence
#
accuracy = clf.score(X_test, y_test)

print("accuracy: ", accuracy)

# let's create a new data set and predict
# benign vs. malig. (keeping in mind that
# we've dropped the id and the class columns)
#
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1) # to avoid deprecation warning

prediction = clf.predict(example_measures)

print("prediction#1: ", prediction)

# let's try predicting two data points:

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])

# to avoid deprecation warning
#
# also, if the # of incoming data points to test for
# varies widely week by week; we don't want to hard
# code the shape size, so use len();
example_measures = example_measures.reshape(len(example_measures), -1) 


prediction = clf.predict(example_measures)

print("prediction#2: ", prediction)

# Euclidean Distance example:
# 2 dimensions: 
# q = (1, 3)  
# p = (2, 5)
#
# Euclidean Distance = Sqrt( (1 - 2)**2 + (3 - 5)**2 )
# Sqrt(1 + 4) = Sqrt(5)
#

plot1 = [1, 3]
plot2 = [2, 5]

euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )

print("euclidean_distance: ", euclidean_distance)

# now hand define out features class and label class
# we have two classes and their features
#
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]] }

new_features = [5,7]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], s=100, color=i)
#plt.show()

result, confidence = k_nearest_neighbors(dataset, new_features, k=3)

print("result: ", result, "confidence#2: ", confidence)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.show()


# Sample Output:
#
"""
Output comparing hand-written KNN vs Scikit's KNN():

Accuracy fr Func x25:  0.9568345323741008
Accuracy fr Func x25:  0.9640287769784173
Accuracy fr Func x25:  0.9784172661870504
Accuracy fr Func x25:  0.9784172661870504
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9856115107913669
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9640287769784173
Accuracy fr Func x25:  0.9784172661870504
Accuracy fr Func x25:  0.9784172661870504
Accuracy fr Func x25:  0.9640287769784173
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9640287769784173
Accuracy fr Func x25:  0.9928057553956835
Accuracy fr Func x25:  1.0
Accuracy fr Func x25:  0.9568345323741008
Accuracy fr Func x25:  0.935251798561151
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9424460431654677
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9712230215827338
Accuracy fr Func x25:  0.9568345323741008
Accuracy fr Func x25:  0.9856115107913669
Avg. Accuracy: fr Func  0.9700719424460429
Accuracy fr KNN() x25:  0.964285714286
Accuracy fr KNN() x25:  0.964285714286
Accuracy fr KNN() x25:  0.971428571429
Accuracy fr KNN() x25:  0.964285714286
Accuracy fr KNN() x25:  0.964285714286
Accuracy fr KNN() x25:  0.957142857143
Accuracy fr KNN() x25:  0.971428571429
Accuracy fr KNN() x25:  0.964285714286
Accuracy fr KNN() x25:  0.971428571429
Accuracy fr KNN() x25:  0.971428571429
Accuracy fr KNN() x25:  0.985714285714
Accuracy fr KNN() x25:  0.971428571429
Accuracy fr KNN() x25:  0.978571428571
Accuracy fr KNN() x25:  0.95
Accuracy fr KNN() x25:  0.971428571429
Accuracy fr KNN() x25:  0.985714285714
Accuracy fr KNN() x25:  0.964285714286
Accuracy fr KNN() x25:  0.978571428571
Accuracy fr KNN() x25:  0.985714285714
Accuracy fr KNN() x25:  0.978571428571
Accuracy fr KNN() x25:  0.964285714286
Accuracy fr KNN() x25:  0.978571428571
Accuracy fr KNN() x25:  0.957142857143
Accuracy fr KNN() x25:  0.978571428571
Accuracy fr KNN() x25:  0.985714285714
Avg. Accuracy: fr neighbors.KNeighborsClassifier() x25 0.971142857143
accuracy:  0.992857142857
prediction#1:  [2]
prediction#2:  [2 2]
euclidean_distance:  2.23606797749979
result:  r confidence#2:  1.0
"""
