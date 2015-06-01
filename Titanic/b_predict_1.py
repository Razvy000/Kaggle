import csv as csv 
import numpy as np

# read test file
test_file = open('csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# open or create a new file
prediction_file = open("output/genderbasedmodel.csv", "w+b")
prediction_file_object = csv.writer(prediction_file)

# simple predict based on gender
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()