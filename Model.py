import pandas as pd
import numpy as np
import sklearn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc , roc_curve, accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from over_under_sampling import over_under_sampling


X = np.load('C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/X.npy')
y = np.load('C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/y.npy')


# Using the Cross validation 

k_fold = KFold(n_splits=2, shuffle=True, random_state=0)

# different models

# Random Forest
#model = RandomForestClassifier(n_estimators= 300, random_state = 20)

# SVM
# model = SVC(C=1.0, kernel='linear', degree=3, coef0=0.0, shrinking=True,
#            probability=True,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
#            decision_function_shape='ovo', random_state=None)

#model = SVC(kernel='linear')

# Bagging Regressor
model = sklearn.ensemble.BaggingRegressor(base_estimator=sklearn.ensemble.GradientBoostingRegressor(max_depth=4,
                                            n_estimators=130),
                                          n_estimators=30)


# creating the plot of PR curve and ROC curve with the model above
# plotting pr and roc curve
f, axes = plt.subplots(1, 2, figsize=(10, 5))

# need this initialization to get the PR curve in cross validation 
y_real = []
y_proba = []
y_pred=[]

for i, (train_index, test_index) in enumerate(k_fold.split(X)):
    
    print('i:',i)
    
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
        
    # re-sampling the data to get more balanced dataset    
    Xtrain , ytrain , Xtest, ytest = over_under_sampling (Xtrain,ytrain, Xtest, ytest)
    
    # fitting the model 
    model.fit(Xtrain, ytrain)
    # predicitng the output probability
    #pred_proba = model.predict_proba(Xtest)
    pred_proba = model.predict(Xtest)
    
    # for the sake of ploting the PR Curve and ROC Curve
    # getting the Precision and Recall
    precision, recall, _ = precision_recall_curve(ytest, pred_proba)#[:,1])
    # getting False positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(ytest, pred_proba)#[:, 1])
    
    # accuracy of the model
    yhat=[]
    out_data = model.predict(Xtest)
    # for the bagging Regressor for the the others which has probabilty this would be the predict output
    for i in range(len(out_data)):
        if out_data[i]<=0.5:
            yhat.append(0)
        else:
            yhat.append(1)
    acc = accuracy_score(ytest, yhat)
        
    # pr curve
    lab = 'Fold %d AUC=%.4f ACC=%.4f' % (i+1, auc(recall, precision),acc)
    axes[0].step(recall, precision, label=lab)

    # roc curve
    lab = 'Fold %d AUC=%.4f ACC=%.4f' % (i+1, auc(fpr, tpr), acc)
    axes[1].step(fpr, tpr, label=lab)

    # getting all values of test and probability of cross validation
    y_real.append(ytest)
    y_proba.append(pred_proba)#[:,1])
    y_pred.append(yhat)

    
    
    
# final pr and roc curve

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
y_pred = np.concatenate(y_pred)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
fpr, tpr, thresholds = roc_curve(y_real,  y_proba)
acc = accuracy_score(y_real, y_pred)

# pr curve
lab = 'Overall AUC=%.4f ACC=%.4f' % (auc(recall, precision),acc)
axes[0].step(recall, precision, label=lab, lw=2, color='black')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].legend(loc='lower left', fontsize='small')

#roc curve
lab = 'Overall AUC=%.4f ACC=%.4f' % (auc(fpr, tpr),acc)
axes[1].step(fpr, tpr, label=lab, lw=2, color='black')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower left', fontsize='small')

f.tight_layout()
f.savefig('C:/Users/lab/Desktop/Maryam/My_CV/applied_Jobs/Scotia Bank/GiveMeSomeCredit/Pr_ROC_curve.png')

