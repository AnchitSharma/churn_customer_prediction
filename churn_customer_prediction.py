# We have customer data of TELCO company with several features.
# Now, because lots of customers are leaving this company, so as part of 
# customer retention program we need to predict customer churn before they decide to leave.
# In order to do that we need to use this data and create machine learning model for customer churn prediction

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/Telco-Customer-Churn.csv")
#df.dtypes

def changeColumnsToString(df):
    columnsNames=['Partner','Dependents','PhoneService','MultipleLines',
                  'OnlineSecurity','OnlineBackup','DeviceProtection',
                  'TechSupport','StreamingTV','StreamingMovies',
                  'PaperlessBilling','Churn']
    for col in columnsNames:
        df[col] = df[col].astype('str').str.replace("Yes",
          '1').replace("No", '0').replace('No internet service',
                      '0').replace('No phone service', 0)
        
changeColumnsToString(df)


df['SeniorCitizen'] = df['SeniorCitizen'].astype(bool)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")

# There are some categorical values which can be encoded as numbers, so we will 
# take a look at unique values present as categories and convert these fields as
# category and encode them

print("Payment methods: ", df.PaymentMethod.unique())
print("Contract types: ", df.Contract.unique())
print("Gender: ", df.gender.unique())
print("Senior citizen: ", df.SeniorCitizen.unique())
print("Internet Service Types: ", df.InternetService.unique())


pay_dic = {'Electronic check':1, 'Mailed check':2 ,'Bank transfer (automatic)':3,
 'Credit card (automatic)':4}

cont_dic = {'Month-to-month':1, 'One year':2, 'Two year':3}

gender_dic = {'Female':1, 'Male':2}

inter_dic = {'DSL':1, 'Fiber optic':2, 'No':0}
df.PaymentMethod = df.PaymentMethod.map(pay_dic)
df.Contract = df.Contract.map(cont_dic)
df.gender = df.gender.map(gender_dic)
df.SeniorCitizen  = df.SeniorCitizen.astype('int64')
df.InternetService = df.InternetService.map(inter_dic)

columnsNames = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

for cols in columnsNames:
    df[cols] = pd.to_numeric(df[cols], errors="coerce")
    
df.dtypes
modelData = df.loc[:, df.columns != 'customerID']
modelData.to_csv("modelData.csv", index=False)


# Model Development
modelData = pd.read_csv("modelData.csv")
modelData.isnull().values.any()
modelData.isnull().sum().sum()
modelData[modelData == np.inf] = np.nan
modelData.fillna(modelData.mean(), inplace=True)

X = modelData.iloc[:, :-1]
y = modelData.iloc[:, -1]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Before fiiting data to our model, feature selection is very essential part of model
# development
# Here we are using sklearn's RandomForestClassifier with esemble learning to choose
# most relevent features for our model. it will iteractively select most 
# relevent features and eliminate least relevent features and threshold will be
# median for feature selection.


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(RandomForestClassifier(n_estimators=100,
                                                random_state=37), threshold="median")

select.fit(X_train, y_train)


X_train_s = select.transform(X_train)

# we can see here black colored area shows all those features are relevent and selected
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Index of features')



# we are fitting our training data to LogisticRegression and making prediction on 
# our test data

X_test_s = select.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
lr = LogisticRegression()
lr.fit(X_train_s, y_train)

y_pred = lr.predict(X_test_s)

lr.score(X_test_s, y_test)

cm = confusion_matrix(y_test, y_pred)
cm


























































































































    
    












































