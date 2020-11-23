import pandas as pd
import numpy as np
from detect_outliers import detect_outliers


#reading the training and test data 
train_data = pd.read_csv("C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/GiveMeSomeCredit/cs-training.csv")
test_data = pd.read_csv("C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/GiveMeSomeCredit/cs-test.csv")

# These are the numerical features present in the dataset
Outliers_to_drop = detect_outliers(train_data,2,["RevolvingUtilizationOfUnsecuredLines",
                                            "age",
                                            "NumberOfTime30-59DaysPastDueNotWorse",
                                            "DebtRatio",
                                            "MonthlyIncome",
                                            "NumberOfOpenCreditLinesAndLoans",
                                            "NumberOfTimes90DaysLate",
                                            "NumberRealEstateLoansOrLines",
                                            "NumberOfTime60-89DaysPastDueNotWorse",
                                            "Unnamed: 0",
                                            "NumberOfDependents"])

dataset =  pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)
# groping last values of NumberOfTime30-59DaysPastDueNotWorse feature to reduce the variables
for i in range(len(dataset)):
    if dataset['NumberOfTime30-59DaysPastDueNotWorse'][i] >= 6:
        dataset['NumberOfTime30-59DaysPastDueNotWorse'][i] = 6
        
# groping last values of NumberOfTimes90DaysLate feature to reduce the variables        
for i in range(len(dataset)):
    if dataset.NumberOfTimes90DaysLate[i] >= 5:
        dataset.NumberOfTimes90DaysLate[i] = 5

# groping last values of NumberRealEstateLoansOrLines feature to reduce the variables         
for i in range(len(dataset)):
    if dataset.NumberRealEstateLoansOrLines[i] >= 6:
        dataset.NumberRealEstateLoansOrLines[i] = 6
        
# groping last values of NumberOfTime60-89DaysPastDueNotWorse feature to reduce the variables         
for i in range(len(dataset)):
    if dataset['NumberOfTime60-89DaysPastDueNotWorse'][i] >= 3:
        dataset['NumberOfTime60-89DaysPastDueNotWorse'][i] = 3
        
# groping last values of NumberOfDependents feature to reduce the variables        
for i in range(len(dataset)):
    if dataset.NumberOfDependents[i] >= 4:
        dataset.NumberOfDependents[i] = 4
        
        
dataset.MonthlyIncome.median()
#Fill Embarked nan values of dataset set with the most frequent value
dataset.MonthlyIncome = dataset.MonthlyIncome.fillna(dataset.MonthlyIncome.median())
dataset.NumberOfDependents = dataset.NumberOfDependents.fillna(dataset.NumberOfDependents.median())
dataset = dataset.drop("Unnamed: 0", axis=1)


len_train = len(train_data)
train = dataset[:len_train]
test = dataset[len_train:]


y = train["SeriousDlqin2yrs"]
X = train.drop( "SeriousDlqin2yrs",axis = 1)
# converting to numpy 
y= y.values
X = X.values

np.save('C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/y.npy', y)
np.save('C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/X.npy', X)

