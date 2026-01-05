import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
x = ["won lottery Lottery", "Meet Tomo", "You won", "Meet today"]
labels = ["1","0","1","0"]

vec=CountVectorizer()
x=vec.fit_transform(x)
print(x)
nb=MultinomialNB()
nb.fit(x,labels)
##pred=nb.predict(x)
print(nb.predict(vec.transform(["Meet"])))
