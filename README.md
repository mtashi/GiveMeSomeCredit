## Problem

https://www.kaggle.com/c/GiveMeSomeCredit

This problem uses the Kaggle dataset about the credit repayment difficulty rates among customers. 

## Datasets
#### SeriousDlqin2yrs:
Person experienced 90 days past due delinquency or worse (Target variable / label).

#### RevolvingUtilizationOfUnsecuredLines:
Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits.

#### age:
Age of borrower in years.

#### NumberOfTime30-59DaysPastDueNotWorse:
Number of times borrower has been 30-59 days past due but no worse in the last 2 years.

#### DebtRatio:
Monthly debt payments, alimony,living costs divided by monthy gross income.

#### MonthlyIncome:
Monthly income.

#### NumberOfOpenCreditLinesAndLoans:
Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards).

#### NumberOfTimes90DaysLate:
Number of times borrower has been 90 days or more past due.

#### NumberRealEstateLoansOrLines:
Number of mortgage and real estate loans including home equity lines of credit.

#### NumberOfTime60-89DaysPastDueNotWorse:
Number of times borrower has been 60-89 days past due but no worse in the last 2 years.

#### NumberOfDependents:
Number of dependents in family excluding themselves (spouse, children etc.)

## Approach
This problem can be treated as a binary calssification where the target is SeriousDlqin2yrs with either 0 or 1 as its values. I mostly used SKlearn library on cpu. 
- First, I analyzed and clean the dataset. There are two columns that have null values. I filled out null values with most occurring values in the column. Also, to reduce the prediction variables and boost the predictive capacity I grouped some of the values in a few features. 
- Then, I used a few methods such as BaggingRegressor, RandomnForestclassifiers, ... to create a model. Since the dataset is very skewed therefore, I resampled the data to balance the dataset. I could not achieve notably higher accuracy, but I got a better PR curve representation. To test the effectiveness of the model I used cross validation. 
- I chose the ROC and PR curves to depict the skill of the model. The highest accuracy achieved was about 79 percent with sklearn.ensemble.BaggingRegressor model.
- The results are shown in the image below. Labels show the area under the curve as well as the accuracy of the model for each fold in the cross validation method. 
- I attached the ipython which describes every step I used in data cleaning and modeling.
- For further expermiment I would consider using XGBOOST library and NN methods. 

![PR_ROC](/PR_ROC_curve.png)
