import pandas as pd
import numpy as np

df = pd.read_csv('csv/train.csv', header=0)

df

df.head(3)

df.tail(3)

type(df)

df.dtypes

df.info()

df.describe()

df['Age'][0:10]

df.Age[0:10]

type(df['Age'])

df['Age'].mean()

df[['Sex', 'Pclass', 'Age']]

df[df['Age']> 60]

df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]

df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

for i in range(1,4):
        print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])

import pylab as P

df['Age'].hist()

P.show()
 
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()

df['Gender'] = 4

df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df.head(3)

median_ages = np.zeros((2,3))

median_ages

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages

df["AgeFill"]=df['Age']

df.head

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]
                

df[df['Age'].isnull()]

df[df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df.describe

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

df.hist

df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()

df['Age*Class'].dropna().hist(bins=100, range=(0,300), alpha = .5)
P.show()

df.dtypes

df['FamilySize'].dropna().hist(bins=15, range=(0,15), alpha = .5)
P.show()

df.dtypes[df.dtypes.map(lambda x: x=='object')]

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 

# get numpy array from pandas
train_data = df.values

train_data

import csv as csv 
csv_file = csv.reader(open('csv/train.csv', 'rb'))
header = csv_file.next()
data = []
for row in csv_file:
    data.append(row)    
data = np.array(data)

# original data
data

# save history
import readline
readline.write_history_file('history')

# remove names
df = df.drop(['PassengerId'], axis = 1)

# fill nan
df[df.Age.isnull()] = df.Age.mean()
df = df.drop(['Age'], axis = 1)

# slow typing fix
#%pylab