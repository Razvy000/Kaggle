import pandas as pd
import numpy as np


# process data
def process_data(path):
    df = pd.read_csv(path, header=0)

    # make a new int Gender column
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # make a new AgeFill column without nan
    # fill missing ages based on gender and pclass
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & \
                                  (df['Pclass'] == j+1)]['Age'].dropna().median()
    # copy
    df["AgeFill"]=df['Age']

    # fill
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                    'AgeFill'] = median_ages[i,j]
                    
    # fill Fare
    df[df.Fare.isnull()] = df.Fare.mean
    
    # make a new AgeIsNull to keep indication if it was originally a missing value
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

    # make a new FamilySize column derived
    # a = df['SibSp'] + df['Parch']
    # a = df.SibSp + df.Parch
    #df['FamilySize'] = df.SibSp + df.Parch

    # make a new AgeMultWithClass column derived
    # df['Age*Class'] = df.AgeFill * df.Pclass

    # drop non int64, int32, float64 
    df = df.drop(['PassengerId', 'Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 

    
    
    return df


# get train and test

# convert from pandas to numpy

#df1 = process_data('csv/train.csv')
#train_data = df1.values

df2 = process_data('csv/test.csv')
test_data = df2.values




# import the random forest package
#from sklearn.ensemble import RandomForestClassifier 

# create the random forest object which will include all the parameters for the fit
#forest = RandomForestClassifier(n_estimators = 100)

# fit the training data to the Survived labels and create the decision trees
#forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# take the same decision trees and run it on the test data
#output = forest.predict(test_data)
