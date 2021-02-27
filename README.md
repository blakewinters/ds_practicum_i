# Practicum I: Prosper Loan Analysis

## Project Summary

Prosper is a Peer-to-Peer (P2P) lending platform that allows both individual and institutional investors to provide loans to other individuals. This marketplace is unique in its ability to circumvent traditional banks or money lending companies in providing needed funds to individuals. This has the benefit of giving individuals with low credit history or other traditionally negative financial characteristics the opportunity to receive a loan.
  
In the following study, I will be analyzing just over a million loans ranging from 2005 to present. The goal of the project is to predict which loans will provide the best investment opportunities using defaults as the target variable. Due to the binary nature of default status, this will be a classification exercise. The task included acquiring and joining together multiple datasets, performing Exploratory Data Analysis (EDA), cleaning the data, selecting features, and finally building and executing predictive models. 


## Data

3 data sets were merged into one clean file for analysis:
  1. Loans files
        - 9 files, 22 columns, 1,329,028 rows, 277 MB
        - Primary data set consisting of several loan files. 
        - Key data points include loan size, loan status, and borrower rate.
        - These files were manually unzipped, then read as a dataframe using a for loop. 
  2. Listings files
        - 9 files, 448 columns, 2,090,506 rows, 8 GB
        - Contains data about the loan at the time of listing on the site.
        - Key data points include borrower income, credit rating, employment status, and job category.
        - These details are crucial to the prediction of loan outcomes.
  3. Master file
        - 1 file, 88 columns, 50,717,253 rows, 34 GB
        - While this file contains details at the loan and listing level, it alco contains line items for every monthly update.
        - Because of this, the file was too much to process in full, and it was stripped down to just mapping fields to join Loans and Listings as well as key additional columns unique to this file. 
        - Even when slimming down the file significantly, it still was too much for my machine to process using Pandas. 
        - I used the Dask library to process the file, which allows for parallel computing of large files, but any updates still took a significant amount of time.

### Merged File



### Libraries

Some basic Git commands are:
```
#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn import pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import seaborn as sns
import io
import glob
from IPython.display import display
from datetime import datetime
import statsmodels.api as sm
import dask.dataframe as dd
from dask.distributed import Client, progress
import tpot
from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline
%matplotlib inline
```

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
