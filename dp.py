# Ibrahim Said

# ASOS - Churn Modelling

# import pandas and numpy
import numpy as np
import pandas as pd
import os

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





# Encoding country names and gender as unique integers
encoder = preprocessing.LabelEncoder()
cusData["shippingCountry"] = encoder.fit_transform(cusData["shippingCountry"])
cusData["gender"] = encoder.fit_transform(cusData["gender"])

# Parsing dates
cusData["dateCreated"] = pd.to_datetime(cusData["dateCreated"])
retData["returnsDate"] = pd.to_datetime(retData["returnsDate"])

# Number of purchases and number of returns and total sales(excluding returns atm) is calculated for each customer.

recCount = recData[["customerId","productId"]].groupby("customerId").agg("count")
recSum = recData[["customerId","productId"]].groupby("customerId").agg("sum")
retCount = retData[["customerId","productId"]].groupby("customerId").agg("count")

recCount["customerId"] = recCount.index
recSum["customerId"] = recSum.index
retCount["customerId"] = retCount.index

#Adding columns receipt count, receipt sum and return count
cusData = pd.merge(cusData,recCount,on="customerId")
cusData = pd.merge(cusData,recSum,on="customerId")
cusData = pd.merge(cusData,retCount,how="left",on="customerId")

#re-labling columns
new_columns = cusData.columns.values
new_columns[-3] = 'NoPurchases'
new_columns[-2] = 'sales'
new_columns[-1] = 'NoReturns'
cusData.columns = new_columns
cusData["NoReturns"] = cusData["NoReturns"].fillna(0)

X = cusData.copy()
y = X.churn
# Dropping columns which will not be used
X.drop(["customerId","dateCreated", "churn"],axis=1,inplace=True)

lr = lm.LinearRegression()
model = lr.fit(X,y)
X = X.as_matrix()
stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True)
y_pred = y.copy()

for ii, jj in stratified_k_fold:
    X_train, X_test = X[ii], X[jj]
    y_train = y[ii]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train,y_train)
    y_pred[jj] = clf.predict(X_test)
print y_pred
print y


# Metrics
fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=2)
print "Accuracy: {}".format(metrics.accuracy_score(y,y_pred))
print "Recall: {}".format(metrics.recall_score(y,y_pred))
print "AUC: {}".format(metrics.roc_auc_score(y, y_pred))
