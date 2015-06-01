import csv as csv 
import numpy as np

csv_file = csv.reader(open('csv/train.csv', 'rb'))
header = csv_file.next()

data = []
for row in csv_file:
    data.append(row)
    
data = np.array(data)

# idea have a survival table
# In the case of a model that uses gender, class, and ticket price,
# will need an array of 2x3x4 
# ( [female/male] , [1st / 2nd / 3rd class], [4 bins of prices] ).

# bining prices

# add a ceiling
fare_ceiling = 40

# (index 9) is Fare
bigger_than_fare = data[0::, 9].astype(np.float) >= fare_ceiling
data[bigger_than_fare] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

# I know there are 3 classes: 1st, 2nd, 3rd
number_of_classes = 3

# but better to calculate from data (index 2) is Pclass 
number_of_classes = len(np.unique(data[0::, 2]))

# initialize the survival table with all zeroes
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

# PassengerId = 0
# Survived = 1    
# Pclass = 2    
# Name = 3    
# Sex = 4    
# Age = 5
# SibSp = 6
# Parch    = 7
# Ticket = 8
# Fare = 9
# Cabin = 10
# Embarked = 11
# loop through each variable and find passengers that agreee
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        women_only_stats = data[
            (data[0::, 4] == "female") &  # Sex
            (data[0::, 2].astype(np.float) == i + 1) &  # Pclass
            (data[0::, 9].astype(np.float) >= j * fare_bracket_size) &  # Fare bucket
            (data[0::, 9].astype(np.float) < (j + 1) * fare_bracket_size)
            ,
            1]  # Survived
         
        men_only_stats = data[
            (data[0::, 4] != "female") &  # Sex
            (data[0::, 2].astype(np.float) == i + 1) &  # Pclass
            (data[0::, 9].astype(np.float) >= j * fare_bracket_size) &  # Fare bucket
            (data[0::, 9].astype(np.float) < (j + 1) * fare_bracket_size)
            ,
            1]  # Survived
            
        # compute statistics
        # will have nan if there are no elements in that subcategory
        survival_table[0, i, j] = np.mean(women_only_stats.astype(np.float)) 
        survival_table[1, i, j] = np.mean(men_only_stats.astype(np.float))
        
# fix nans
survival_table[ survival_table != survival_table ] = 0

# for this model we assume Probability > 0.5 means it survives
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 

# predict based on gender, passenger class and fare bin
test_file = open('csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
predictions_file = open("output/genderclassmodel_2.csv", "w+b")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

# go through tests
for row in test_file_object:  # We are going to loop
                                              # through each passenger
                                              # in the test set                     
    for j in xrange(number_of_price_brackets):  # For each passenger we
                                              # loop thro each price bin
        try:  # Some passengers have no
                                                  # Fare data so try to make
          row[8] = float(row[8])  # a float
        except:  # If fails: no data, so 
          bin_fare = 3 - float(row[1])  # bin the fare according Pclass
          break  # Break from the loop
        if row[8] > fare_ceiling:  # If there is data see if
                                                  # it is greater than fare
                                                  # ceiling we set earlier
          bin_fare = number_of_price_brackets - 1  # If so set to highest bin
          break  # And then break loop
        if row[8] >= j * fare_bracket_size and row[8] < (j + 1) * fare_bracket_size:             
                                                  # If passed these tests 
                                                  # then loop through each bin 
            bin_fare = j                            # then assign index

      
    if row[3] == 'female':  # If the passenger is female
        p.writerow([row[0], "%d" % int(survival_table[0, float(row[1]) - 1, bin_fare])])
    else:  # else if male
        p.writerow([row[0], "%d" % int(survival_table[1, float(row[1]) - 1, bin_fare])])
        
# Close out the files.
test_file.close() 
predictions_file.close()