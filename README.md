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

### Libraries

The primary library for cleaning and structuring data was Pandas. 

Sci-kit-learn (sklearn) was used for the majority of modeling. Even TPOT, which is an AutoML library, utilizes sklearn on the backend. 

MatPlotLib was the library used for visualizations.

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

The Master file was broken down to just LoanID, the Primary Key for the Loan files, and ListingNumber, the Primary Key for the Listing files. 

Dask commands work almost the same as Pandas, but they require the Dataframe to be computed once various commands have been applied. The compute process takes several minutes each time, so computations must be used sparingly. 

```
#Setting client to view progress of each command
client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
client
```
![image](https://user-images.githubusercontent.com/1417344/109394399-0fed0400-78e4-11eb-89fe-1af7c4f66adc.png)

The Client library allows the computations used in Dask to be monitored:

![image](https://user-images.githubusercontent.com/1417344/109394442-3f9c0c00-78e4-11eb-86e3-1225c3b9aa17.png)

For the Listings files, once they were concatenated into a single Dataframe, the majority of the columns were dropped as they were not providing any value. The remaining trimmed down fields allowed the file to be more manageable. In addition, I excluded any listings that were not fulfilled, as they would not exist as eventual Loans.

Finally, I joined the Loans files to the Master file using LoanID, then joined the Listings file to that Dataframe using ListingNumber.

### Cleaning

Several unnecessary columns were dropped, which helped the processing time in the following step. Current and Cancelled Loans were removed from the dataset as well, so that the final analysis would only be performed on Completed or Defaulted Loans. 

A field was added called "Default_Flag", which would serve as the dependent variable for the modeling portion. 0's refer to defaulted loans, whereas 1's represent loans that were either defaulted or charged-off.
```
#Combined Charge Offs and Defaults into one Value
df_master['Default_Flag'] = 0
df_master.loc[((df_master['loan_status'] == 2) | (df_master['loan_status'] == 3)  ) , 'Default_Flag'] = 1
```

Various files previously coded as Boolean TRUE/FALSE were converted to binary 0/1 values. 
```
#Convert fields from Boolean to int
df_analysis['group_indicator'] = (df_analysis['group_indicator'] == 'TRUE').astype(int)
```

Quite a few columns had NA values, and those were adjusted in a variety of ways, including 0 for data such as delinquencies that were not positive, as well as median values for fields like monthly debt. 
```
#Replace Delinquency and Loan data (Prosper internal) Nulls with 0
zero_cols = ['prior_prosper_loans_principal_borrowed','prior_prosper_loans_principal_outstanding',
             'prior_prosper_loans_balance_outstanding','prior_prosper_loans_cycles_billed',
             'prior_prosper_loans_ontime_payments','prior_prosper_loans_late_cycles',
             'prior_prosper_loans_late_payments_one_month_plus','max_prior_prosper_loan','min_prior_prosper_loan',
             'prior_prosper_loan_earliest_pay_off','prior_prosper_loans31dpd','prior_prosper_loans61dpd',
             'current_delinquencies','delinquencies_last7_years','now_delinquent_derog','was_delinquent_derog',
             'delinquencies_over30_days','delinquencies_over60_days','delinquencies_over90_days']

for col in zero_cols:
    df[col].fillna(0, inplace=True)
    
#Replace public credit data (external) with Median
med_cols = ['monthly_debt','funding_threshold','public_records_last10_years','public_records_last12_months',
            'credit_lines_last7_years','inquiries_last6_months','current_credit_lines','open_credit_lines',
            'bankcard_utilization','total_open_revolving_accounts','installment_balance','real_estate_balance',
            'revolving_balance','real_estate_payment','revolving_available_percent','total_inquiries','total_trade_items',
            'satisfactory_accounts']

for col in med_cols:
    df[col].fillna(df[col].median(), inplace=True)
     
    
#Replace amount_delinquent with median where loan applicant has had previous delinquencies, otherwise 0
#Calculate median of delinquency amount
ad_med = df['amount_delinquent'].median()

#Apply median where first_recorded_credit_line is null and applicant has credit lines
df.loc[(df['amount_delinquent'].isnull()) & ((df['current_delinquencies'] > 0) | (df['delinquencies_last7_years'] > 0)), 'amount_delinquent'] = ad_med

#Replace remaining NAs for amount_delinquent with 0
df['amount_delinquent'].fillna(0, inplace=True)
```

Various fields such as credit risk or income range were manually encoded, as opposed to performing Standard or One-Hot-Encoding, because their values are linear in nature.
```
#Custom labels for Prosper rating
#Note: HR = High Risk
custom_mapping = {'HR': 1, 'E': 2, 'D': 3, 'C': 4, 'B': 5, 'A': 6, 'AA': 7}

df['prosper_rating_mod'] = df['prosper_rating'].map(custom_mapping)
```


## EDA

Various EDA techniques were used, primarily around correlation and bar charts to display default rates. The correlation results allowed for the dataset to be reduced by removing fields with virtually no relationship with default rate, as well as several fields that were colinear.
```
#Correlation Heatmap
_ = sns.heatmap(df.corr())
```
![image](https://user-images.githubusercontent.com/1417344/109395517-0d8da880-78ea-11eb-80c9-eb9484f04926.png)

The dataset is skewed toward completed loans, which had to be taken into account when building the models.
![image](https://user-images.githubusercontent.com/1417344/109395525-17171080-78ea-11eb-8717-bddb18b3438f.png)

Individuals with lower credit scores defaulted at a higher rate.
![image](https://user-images.githubusercontent.com/1417344/109395546-2bf3a400-78ea-11eb-8088-be45cd2e45c3.png)

Similarly, individuals with lower incomes or $0 incomes defaulted at a higher rate.
![image](https://user-images.githubusercontent.com/1417344/109395572-475eaf00-78ea-11eb-8b82-3ca6cedb9c44.png)

## Feature Engineering

### One Hot Encoding

One Hot Encoding was used to convert important fields like Employment Status to binary columns, which could then be used in a model. Standard encoding would not suffice here because the values could be considered more important than each other. 
```
#Use One Hot Encoding for object columns
#Employment Status Description
dum = pd.get_dummies(df_clean['employment_status_description'], prefix='Employment')
df_clean = df_clean.join(dum)
```

### Scaling
Now that the columns were all in numeric format, I used MinMaxScaler to convert them to a consistent 0-1 scale. This ensured that fields with wide ranges would not overly skew their impacts in a model. I also tested StandardScaler, but achieved worse results.
```
#Scale values for better comparison in model
#scaler = preprocessing.StandardScaler()
scaler = preprocessing.MinMaxScaler()

scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)
```

### RFE
Recursive Feature Elimination (RFE) was used to identify the most important columns. I tested various levels of columns to select, ending on the 47 most important features. This provided the best performance while still improving performance compared to using the entire dataset.
```
rfe = rfe.fit(X_scaled, y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#Extract most important features
rfe_cols = rfe.get_support(1) 
```

### PCA

I used Principal Component Analysis to convert the values down to their principal components, which led to new variables that were the most impactful to the target variable. I tested several numbers of components, and using just the 1 most important principal component led to the best results. 
```
#Convert values to Principal Components
pca = PCA()
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.transform(X_test)
```

## Models

### Goal Accuracy Score

A goal accuracy was calculated based on how soon investments would break even given that the nth loan will eventually default.

I used median rates of return on completed loans as well as median payments received for defaulted loans in order to reach this calculation.
```
#Assuming X amount of loans complete and the X+1 loan defaults, calculate ideal return:
#completed_return * X - defaulted_return * 1 = 0
#2408.0 * X = 6394.0
#X = 2.66
#2.66 completed loans needed for every 1 defaulted loan

acc_ideal_mod = round((def_loans * -1 / comp_loans) / (def_loans * -1 / comp_loans + 1) ,2)

print("Ideal accuracy modified: ", acc_ideal_mod)
```
Using this methodology, the accuracy score to beat was 73%.


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
