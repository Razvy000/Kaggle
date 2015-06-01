import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Object Creation
s = pd.Series([1,3,5,np.nan, 6, 8])

s

# create DataFrame with array, datetime index and label columns
dates = pd.date_range('20130101', periods = 6)

dates

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

df

# create DataFrame by passing dict of object that can be converted to series-like
df2 = pd.DataFrame({ 'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo' })

df2

# check dtypes
df2.dtypes

### Viewing Data

df.head()

df.tail(3)

# display index, columns and underlying numpy data
df.index

df.columns
type(df.columns)

df.values
type(df.values)

# qucik statistics summary of data
df.describe()

# transpose data
df.T

# sort by an axis
df.sort_index(axis=1, ascending=False)

# sort by values
df.sort(columns='B')




#### Selection: use .at .iat .loc .iloc .ix

# Getting
df['A']

# selecting via [], with slices the rows
df[0:3]

df['20130102':'20130104']




### Selection by Label
df.loc[dates[0]]

# selecting on a multi-axis by label
df.loc[:,['A','B']]

# showing label slicing, both endpoints are included
df.loc['20130102':'20130104', ['A', 'B']]

# reduction in the dimensions of the returned object
df.loc['20130102', ['A', 'B']]

# getting a scalar value
df.loc[dates[0],'A']

# fast acces to scalar
df.at[dates[0],'A']




### Selection by Position
df.iloc[3]

# selection by position by integer slices, similar to numpy/python
df.iloc[3:5,0:2]

# selection by position by lists of integer positions, similar to numpy/python
df.iloc[[1,2,3],[0,2]]

# slice rows explicitly
df.iloc[1:3,:]

# slice cols explicitly
df.iloc[:,1:3]

# getting a value explicitly
df.iloc[1,1]

# fast access to a scalar equiv to iloc
df.iat[1,1]




### Boolean Indexing

# use column's value to select
df[df.A > 0]

# a WHERE operation for getting
df[df > 0] # makes NaN values

# using th isin() method for filtering

df2 = df.copy()

df2['E']=['one', 'one','two','three','four','three']

df2

df2[df2['E'].isin(['two','four'])]




### Setting

# setting a new column automatically aligns the data by the indexes

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))

df['F'] = s1

# setting values by label
df.at[dates[0],'A'] = 0

# setting values by position
df.iat[0,1]=0

# setting by assign with a numpy array
df.loc[:,'D'] = np.array([5]*len(df))

# a WHERE operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2;
df2




### Missing Data

# reindex allows you to change/add/delete index on a specified axi
# return a copy of the data

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
df1


# drop rows that have missing data
df1.dropna(how='any')

# fill in missing data
df1.fillna(value=5)

# get boolean mask where values are nan
pd.isnull(df1)




### Operations

# Stats

# descriptive statistic
df.mean()

# same op on the other axis
df.mean(1)

# operating with objects that have different dimensionality and need alignment
# in addition, pandas automatically broadcasts along the specified dimension
s = pd.Series([1,3,4,np.nan,6,8],index=dates).shift(2)

df.sub(s, axis='index')

# Apply
df.apply(np.cumsum)

df.apply(lambda x : x.max() - x.min())

# Histogramming
s = pd.Series(np.random.randint(0,7,size=10))
s
s.value_counts()

# String Methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

s.str.lower()
 
 
 
### Merge

# concat
df = pd.DataFame(np.random.randn(10,4))
df
# break into pieces
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# join
# sql style merges

left = pd.DataFrame({'key' : ['foo', 'foo'], 'lval':[1,2]})

right = pd.DataFrame({'key' : ['foo', 'foo'], 'rval':[4,5]})

pd.merge(left, right, on='key')
 
# append
df = pd.DataFrame(np.random.randn(8,4), columns=['A', 'B', 'C', 'D'])

s = df.iloc[3]

df.append(s, ignore_index=True)

# grouping
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                           'foo', 'bar', 'foo', 'foo'],
                    'B' : ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                    'C' : np.random.randn(8),
                    'D' : np.random.randn(8)})
                    
df.groupby('A').sum()
df.groupby(['A','B']).sum()



### Reshaping

# stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                        'foo', 'foo', 'qux', 'qux'],
                        ['one', 'two', 'one', 'two',
                        'one', 'two', 'one', 'two']]))
                        
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

df2 = df[:4]

df2

# stack compresses a level in the DataFrame's columns
stacked = df2.stack()

# the inverse op is unstack, default unstack last level
stacked.unstack()



### Pivot Tables
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                        'B' : ['A', 'B', 'C'] * 4,
                        'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                        'D' : np.random.randn(12),
                        'E' : np.random.randn(12)})
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])



### Time Series
# pandas has simple, powerful, and efficient functionality for performing resampling operations 
# during frequency conversion (e.g., converting secondly data into 5-minutely data)
# this is extremely common in, but not limited to, financial applications

rng = pd.date_range('1/1/2012', periods=100, freq='S')

ts = pd.Series(np.random.randn(len(rng)), rng)

ts_utc = ts.tz_localize('UTC')

# convert to another time zone
ts_utc.tz_convert('US/Eastern')

# convert between time span representations
rng = pd.date_range('1/1/2012', periods=5, freq='M')

ts = pd.Series(np.random.randn(len(rng)), index=rng)

ts

ps = ts.to_period()

ps.to_timestamp()



### Categorical

# can include categorical data in DataFrame
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")

# assigning to Series.cat.categories is inplace!
df["grade"].cat.categories = ["very good", "good", "very bad"]

# reorder the categories and simultaneously add the missing categories
# methods under Series .cat return a new Series per default
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df["grade"]

# sorting is per order in the categories, not lexical order
df.sort("grade")

# grouping by a categorical column shows also empty categories
df.groupby("grade").size()



### Plotting
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()


# on DataFrame, plot() is a convenience to plot all of the columns with labels
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                    columns=['A', 'B', 'C', 'D'])
                    
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')




### Getting Data In/Out

# csv
# write
df.to_csv('foo.csv')
# read
pd.read_csv('foo.csv')

# hdf5
#write
df.to_hdf('foo.h5','df')
# read
pd.read_hdf('foo.h5','df')

#excell
#write
df.to_excel('foo.xlsx', sheet_name='Sheet1')
#read
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

