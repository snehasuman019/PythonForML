import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Data Preprocessing
dataset=pd.read_csv(("E:\\D drive\\SEM 5\\234\\Market_Basket_Optimisation.csv"),encoding = "Latin1")
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
print("******",transactions)
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift =3, min_length =2, max_length = 2)
results = list(rules)
print("%%%%%%%%%%%%",results)
