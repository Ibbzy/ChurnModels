# Ibrahim Said

# ASOS - Churn Modelling

# import pandas and numpy
import numpy as np
import pandas as pd

# import models from sci-kit learn
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Part I - ETL - Data processing
# Clarification: cus = customer, rec = receipt, ret = return

# Defining headers
cusHeader = ["customerId","gender","shippingCountry","dateCreated","yearOfBirth","premier","churnlabel"]
recHeader = ["customerId","productId","divisionId","discountDesc","itemQty","signalDate","receiptId","price"]
retHeader = ["customerId","productId","divisionId","itemQty","returnsDate","receiptId","returnId","returnAction","returnReason"]


# Reading training datasets
cusData = pd.read_csv("train/customer.txt", sep='\t', lineterminator='\n',names=cusHeader)
recData = pd.read_csv("train/receipts.txt", sep='\t', lineterminator='\n',names=recHeader)
retData = pd.read_csv("train/returns.txt", sep='\t', lineterminator='\n',names=retHeader)

# Parsing Dates
cusData["dateCreated"] = pd.to_datetime(cusData["dateCreated"])
retData["returnsDate"] = pd.to_datetime(retData["returnsDate"])

# Number of purchases and number of returns and total sales(excluding returns atm) is calculated for each customer.

recCount = recData[["customerId","productId"]].groupby("customerId").agg("count")
recSum = recData[["customerId","productId"]].groupby("customerId").agg("sum")
retCount = retData[["customerId","productId"]].groupby("customerId").agg("count")

recCount["customerId"] = recGroup.index
recSum["customerId"] = recGroup.index
retCount["customerId"] = retGroup.index

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
