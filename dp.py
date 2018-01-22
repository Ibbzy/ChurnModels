# Ibrahim Said

# ASOS - Churn Modelling

# import pandas and numpy
import numpy as np
import pandas as pd
import os
import xgboost as xgb

# import models from sci-kit learn

from sklearn import cross_validation
from sklearn import tree
from sklearn import linear_model as lm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# Part I - ETL - Data processing
# Clarification: cus = customer, rec = receipt, ret = return

# Defining headers
cusHeader = ["customerId","gender","shippingCountry","dateCreated","yearOfBirth","premier","churn"]
recHeader = ["customerId","productId","divisionId","discountDesc","itemQty","signalDate","receiptId","price"]
retHeader = ["customerId","productId","divisionId","itemQty","returnsDate","receiptId","returnId","returnAction","returnReason"]


# Reading training datasets
cusData = pd.read_csv("train/customer.txt", sep='\t', lineterminator='\n',names=cusHeader)
recData = pd.read_csv("train/receipts.txt", sep='\t', lineterminator='\n',names=recHeader)
retData = pd.read_csv("train/returns.txt", sep='\t', lineterminator='\n',names=retHeader)

# Reading holdout datasets
cusDatah = pd.read_csv("holdout/customer.txt", sep='\t', lineterminator='\n',names=cusHeader)
recDatah = pd.read_csv("holdout/receipts.txt", sep='\t', lineterminator='\n',names=recHeader)
retDatah = pd.read_csv("holdout/returns.txt", sep='\t', lineterminator='\n',names=retHeader)

#



def prep(cusData,recData,retData,tr=True):

    # Encoding country names and gender as unique integers
    encoder = preprocessing.LabelEncoder()
    cusData["shippingCountry"] = encoder.fit_transform(cusData["shippingCountry"])
    cusData["gender"] = encoder.fit_transform(cusData["gender"])

    # Parsing dates
    cusData["dateCreated"] = pd.to_datetime(cusData["dateCreated"])
    retData["returnsDate"] = pd.to_datetime(retData["returnsDate"])
    recData.signalDate = pd.to_datetime(recData.signalDate.astype(str), format='%Y%m%d')
    recData.signalDate = (max(recData.signalDate)-recData.signalDate).dt.days

    # Number of purchases and number of returns
    # and total sales(excluding returns atm) is calculated for each customer.

    recCount = recData[["customerId","productId"]].groupby("customerId").agg("count")
    recSum = recData[["customerId","price"]].groupby("customerId").agg("sum")
    recItems = recData[["customerId","itemQty"]].groupby("customerId").agg("sum")
    retCount = retData[["customerId","productId"]].groupby("customerId").agg("count")
    recDate = recData[['customerId','signalDate']].groupby("customerId")
    recVar = recDate.var(ddof=1).fillna(0)
    recLast = recDate.min()


    #Fixing indices
    recCount["customerId"] = recCount.index
    recSum["customerId"] = recSum.index
    recItems["customerId"] = recItems.index
    retCount["customerId"] = retCount.index
    recVar["customerId"] = recVar.index
    recLast["customerId"] = recLast.index
    df = cusData.copy()

    #Adding columns receipt count, receipt sum and return count
    df = pd.merge(df,recLast,how="left",on="customerId")
    df = pd.merge(df,recItems,how="left",on="customerId")
    df = pd.merge(df,recSum,how="left",on="customerId")
    df = pd.merge(df,retCount,how="left",on="customerId")
    df = pd.merge(df,recVar,how="left",on="customerId")

    #re-labling columns
    new_columns = df.columns.values
    new_columns[-5] = 'time since latest purchase'
    new_columns[-4] = 'NoPurchases'
    new_columns[-3] = 'sales'
    new_columns[-2] = 'NoReturns'
    new_columns[-1] = 'var of dates'
    df.columns = new_columns
    df["NoReturns"] = df["NoReturns"].fillna(0)
    df["NoItems"] = df["NoPurchases"].sub(df["NoReturns"])


    X = df.copy()
    if tr == True:
        y = X.churn

        # Dropping columns which will not be used

        X.drop(["customerId","dateCreated", "churn","NoReturns"],axis=1,inplace=True)

        return X,y
    else:
        X.drop(["customerId","dateCreated", "NoReturns"],axis=1,inplace=True)
        return X

X,y = prep(cusData,recData,retData,tr=True)
Xh = prep(cusDatah,recDatah,retDatah, tr=False)

# Creating a simple logistic regression model



def train(X,y):
    XX = X.as_matrix()
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True)
    y = y.replace(2,0)
    y_pred1 = y.copy()
    y_pred2 = y.copy()




    for ii, jj in stratified_k_fold:
        X_train, X_test = XX[ii], XX[jj]
        y_train = y[ii]

        clf1 = xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05)
        clf2 = lm.LogisticRegression()


        clf1.fit(X_train,y_train)
        y_pred1[jj] = clf1.predict(X_test)
        clf2.fit(X_train,y_train)
        y_pred2[jj] = clf2.predict(X_test)


    # Metrics
    #fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=2)

    print "precision for xgboost: {}".format(metrics.precision_score(y,y_pred1))
    print "precision for logistic regression: {}".format(metrics.precision_score(y,y_pred2))
    print "Recall for xgboost: {}".format(metrics.recall_score(y,y_pred1))
    print "Recall for logistic regression: {}".format(metrics.recall_score(y,y_pred2))
    print "AUC for xgboost: {}".format(metrics.roc_auc_score(y, y_pred1))
    print "AUC for logistic regression: {}".format(metrics.roc_auc_score(y, y_pred2))

    return clf1,clf2

xgb,logreg = train(X,y)

#predictions

Xh.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
print logreg.predict(Xh)
print xgb.predict(Xh)
