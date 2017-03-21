# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:11:01 2016

@author: Michael O Sullivan
Alot of the code is commented due to test purposes.
Each algorithm is broken up so uncomment the section you wish to test.
"""
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import tree
from sklearn import naive_bayes
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
import matplotlib.pyplot as plt



aDTree = []
aRbfSvm = []
aNearestN = []
aRandomForest = []
npNaiveBayes = []
# Function will run each of the classifiers using cross fold validation
def runClassifiers(data, target):
    
    dTree = tree.DecisionTreeClassifier()
    scores = cross_validation.cross_val_score(dTree, data, target, cv=10)
    aDTree.append(scores.mean())
    print "Tree : ", scores.mean()
    
    rbfSvm = SVC()
    scores = cross_validation.cross_val_score(rbfSvm, data, target, cv=10)
    aRbfSvm.append(scores.mean())
    print "SVM : ", scores.mean()
    
    nearestN = KNeighborsClassifier()
    scores = cross_validation.cross_val_score(nearestN, data, target, cv=10)
    aNearestN.append(scores.mean())
    print "NNeighbour : ", scores.mean()
    
    randomForest = RandomForestClassifier()
    scores = cross_validation.cross_val_score(randomForest, data, target, cv=10)
    aRandomForest.append(scores.mean())
    print "RForest : ",scores.mean()
    
    nBayes = naive_bayes.GaussianNB()
    scores = cross_validation.cross_val_score(nBayes, data, target, cv=10)
    npNaiveBayes.append(scores.mean())
    print "Naive Bayes : ",scores.mean()
    
def optimizationForKNN(data, target):
    
    param_grid = [ {'n_neighbors': range(1, 80),  'p':[1, 2, 3, 4, 5] }  ]
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    clf.fit(data, target)
    
    
    print("Best parameter set found on development set:")
    print(clf.best_params_)

    scores = cross_validation.cross_val_score(clf.best_estimator_, data, target, cv=5)
    print "NNeighbour : ", scores.mean()
    
    
def optimizationForSVM(data, target):
    
    param_grid = [ {'kernel': ['rbf', 'poly', 'linear'],  'C':range(1,15)}  ]
    clf = GridSearchCV(SVC(), param_grid, cv=10)
    clf.fit(data, target)
    
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    
    scores = cross_validation.cross_val_score(clf.best_estimator_, data, target, cv=10)
    print "SVM : ", scores.mean()
#    plt.plot(scores)
#    plt.show()
#    
    

def greedyFeatureSelection(data, target):
       
    while (len(data.columns) > 1):
    #while (len(data.columns) > 30):
        estimator = SVR(kernel="linear")
        scores = cross_validation.cross_val_score(estimator,data, target, cv=10)
        print "Initial Results = ",scores.mean()
        
        selector = RFECV(estimator, cv=3, scoring='mean_squared_error')
        selector.fit(data, target)
        #print "Optimal number of features is ", selector.n_features_
        #print "Optimal number of features is ", selector.
        
        
        print "selector.support_", data.columns.values[selector.support_]
        print selector.ranking_
        #Get the column index of the worst feature
        maxindex = selector.ranking_.argmax()
        print "Worst features is ",[data.columns.values[maxindex]]
        
        df = data[data.columns.values[selector.support_]]
        scores = cross_validation.cross_val_score(estimator, df.values , target, cv=10)
        print "Result after feature selection ",scores.mean()
        data.drop(data.columns.values[maxindex], axis=1, inplace=True)
        runClassifiers(data, target)
        print ""
        print ""
    

def multiclass_classification(data, target):
    print "OneVsRestClassifier"
    print OneVsRestClassifier(LinearSVC(random_state=0)).fit(data, target).predict(data)
  

def convertColumnToBinary(dataframe, columnName):
    
    averageFinalGrade = round(dataframe[[columnName]].mean(),0)
    gradeRow = pd.Series(dataframe[columnName])
    
    #Using binary classification by getting the average for G3
#    gradeRow[gradeRow < averageFinalGrade] = 0
#    averageFinalGrade = round(dataframe[[columnName]].mean(),0)
#    gradeRow = pd.Series(dataframe[columnName])
#    gradeRow[gradeRow < averageFinalGrade] = 0
#    gradeRow[gradeRow >= averageFinalGrade] = 1
    
    #5 Level Classification
    gradeRow[gradeRow < 10]  = 5
    gradeRow[gradeRow >= 16] = 1
    gradeRow[gradeRow >= 14] = 2
    gradeRow[gradeRow >= 12] = 3
    gradeRow[gradeRow >= 10] = 4
    
    dataframe[columnName] = gradeRow


    dataframe['school'] = dataframe['school'].map({'GP': 0, 'MS':1}).astype(int)
    dataframe['sex'] = dataframe['sex'].map({'M': 0, 'F':1}).astype(int)
    dataframe['Pstatus'] = dataframe['Pstatus'].map({'A': 0, 'T':1}).astype(int)
    dataframe['higher'] = dataframe['higher'].map({'no': 0, 'yes':1}).astype(int)
    dataframe['internet'] = dataframe['internet'].map({'no': 0, 'yes':1}).astype(int)
    
    dataframe['address'] = dataframe['address'].map({'U': 0, 'R':1}).astype(int)
    dataframe['famsize'] = dataframe['famsize'].map({'LE3': 0, 'GT3':1}).astype(int)
    dataframe['Mjob'] = dataframe['Mjob'].map({'teacher': 0, 'health':1, 'services':2, 'at_home':3, 'other':4}).astype(int)
    dataframe['Fjob'] = dataframe['Fjob'].map({'teacher': 0, 'health':1, 'services':2, 'at_home':3, 'other':4}).astype(int)
    dataframe['reason'] = dataframe['reason'].map({'home': 0, 'reputation':1, 'course':2, 'other':3}).astype(int)
    dataframe['guardian'] = dataframe['guardian'].map({'mother': 0, 'father':1, 'other':2}).astype(int)
    
    dataframe['schoolsup'] = dataframe['schoolsup'].map({'yes': 0, 'no':1}).astype(int)
    dataframe['famsup'] = dataframe['famsup'].map({'yes': 0, 'no':1}).astype(int)
    dataframe['paid'] = dataframe['paid'].map({'yes': 0, 'no':1}).astype(int)
    dataframe['activities'] = dataframe['activities'].map({'yes': 0, 'no':1}).astype(int)
    dataframe['nursery'] = dataframe['nursery'].map({'yes': 0, 'no':1}).astype(int)
    dataframe['romantic'] = dataframe['romantic'].map({'yes': 0, 'no':1}).astype(int)
    return dataframe
    
portuguese_df = pd.read_excel("C:\Users\micha\Desktop\Semester 7\Machine Learning\Project 2\student-porExcel.xlsx")
math_df = pd.read_excel("C:\Users\micha\Desktop\Semester 7\Machine Learning\Project 2\student-mat.xlsx")

portuguese_df = convertColumnToBinary(portuguese_df, "G3")
math_df = convertColumnToBinary(math_df, "G3")



#-----------------Concatinate both dataframes--------------
#frames = [portuguese_df, math_df]
#concatDF = pd.concat(frames)
#target = concatDF["G3"]
#data = concatDF.drop(["G3"], axis= 1)
#----------------------------------------------------------



#print portuguese_df.describe()
#print portuguese_df.info()


# Next separate portuguese_df class data from the training data
target = portuguese_df["G3"]
data = portuguese_df.drop(["G3"], axis= 1)
#data = data.drop(["G2"], axis= 1)#Drop G2
#data = data.drop(["G1"], axis= 1)#Drop G1





#optimizationForKNN(data, target)
#optimizationForSVM(data, target)

# Next separate portuguese_df class data from the training data
#target = portuguese_df["G3"]
#data = portuguese_df.drop(["G3"], axis= 1)


print "portuguese_df"
#runClassifiers(data, target)

#optimizationForKNN(data, target)
#optimizationForSVM(data, target)




#target = portuguese_df["G3"]
#data = portuguese_df.drop(["G3"], axis= 1)
#print math_df.describe()
#print math_df.info()



print "portuguese_df"
target = portuguese_df["G3"]
data = portuguese_df.drop(["G3"], axis= 1)
runClassifiers(data, target)
#data = data.drop(["G2"], axis= 1)#Drop G2
#data = data.drop(["G1"], axis= 1)#Drop G1
optimizationForKNN(data, target)
optimizationForSVM(data, target)


print "math_df"
target = math_df["G3"]
data = math_df.drop(["G3"], axis= 1)
runClassifiers(data, target)
#data = data.drop(["G2"], axis= 1)#Drop G2
#data = data.drop(["G1"], axis= 1)#Drop G1
#optimizationForKNN(data, target)
#optimizationForSVM(data, target)




#-----------------standarization--------------
#print "Before standarization"
#runClassifiers(data, target)
#scalingObj = preprocessing.StandardScaler()
#standardizedData = scalingObj.fit_transform(data)
#data = pd.DataFrame(standardizedData, columns=data.columns)
#print "After standarization"
#runClassifiers(data, target)
#----------------------------------------------


#-----------------greedyFeatureSelection--------------
#X = data
#y = target
##multiclass_classification(data, target)
#greedyFeatureSelection(data, target)
##plt.plot(aDTree,aRbfSvm,aNearestN, aRandomForest, aNaiveBayes)
#plt.plot(aDTree)
#plt.plot(aRbfSvm)
#plt.plot(aNearestN)
#plt.plot(aRandomForest)
#plt.plot(npNaiveBayes)
#
#plt.legend(['DTree', 'RbfSvm', 'NearestN', 'RandomForest', 'NaiveBayes' ],  loc='upper left', prop={'size':10}, bbox_to_anchor=(1,1))
#plt.tight_layout(pad=7)
##
#plt.show()
#multiclass_classification(data, target)
#----------------------------------------------------------



#Univariate Feature Selection
#Selector_f = SelectPercentile(f_regression, percentile = 25)
#Selector_f.fit(X,y)
#
#for n,s in zip(portuguese_df.dtypes.index, Selector_f.scores_):
#    print 'F Score', s, "for feature ", n

















