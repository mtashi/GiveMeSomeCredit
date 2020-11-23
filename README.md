# Analysis

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
This problem can be treated as a binary calssification where the target is SeriousDlqin2yrs. 
- First, I analyzed the dataset. There are two columns that have null variables. I filled out null values with most occurring value in the column. To reduce the prediction variable and boost the predictive capacity I group some of values in a few columns. 
- Then, I used a few models to fit onto the models. As the dataset is very skewed, I resample the dataset to balance the dataset in combination with cross validation. 
- I chose the ROC and PR curve to show the skill of the model. The highest accuracy achieved was about 68 percent with … model.
- The attached ipython shows every step and its results.

![alt text](C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/GiveMeSomeCredit/Pr_ROC_curve.png)
