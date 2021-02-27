# Practicum I: Prosper Loan Analysis

## Project Summary

Prosper is a Peer-to-Peer (P2P) lending platform that allows both individual and institutional investors to provide loans to other individuals. This marketplace is unique in its ability to circumvent traditional banks or money lending companies in providing needed funds to individuals. This has the benefit of giving individuals with low credit history or other traditionally negative financial characteristics the opportunity to receive a loan.
  
In the following study, I will be analyzing just over a million loans ranging from 2005 to present. The goal of the project is to predict which loans will provide the best investment opportunities using defaults as the target variable. Due to the binary nature of default status, this will be a classification exercise. The task included acquiring and joining together multiple datasets, performing Exploratory Data Analysis (EDA), cleaning the data, selecting features, and finally building and executing predictive models. 

## Process

### Libraries 

Some basic Git commands are:
```
#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import io
import glob
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from datetime import datetime
#from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import tpot
from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline
```

## Data

The primary data set consists of several loan files. These were manually unzipped, then read as a dataframe using a for loop. 

2 additional datasets were merged: 
  1. Listings file

### Merging

### Cleaning

## EDA

## Models

## Conclusion

## References
https://docs.dask.org/en/latest/dataframe.html

https://www.analyticsvidhya.com/blog/2020/02/joins-in-pandas-master-the-different-types-of-joins-in-python/

https://stackoverflow.com/questions/8419564/difference-between-two-dates-in-python

https://datascience.stackexchange.com/questions/70298/labelencoding-selected-columns-in-a-dataframe-using-for-loop

https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html

https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd

https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/

https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

https://stackoverflow.com/questions/57085897/python-logistic-regression-max-iter-parameter-is-reducing-the-accuracy

https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
