# Data Preporcessing

# Importing lib
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
# to show in the variable explorer
df_x = pd.DataFrame(X)

Y = dataset.iloc[:,3].values
# to show in the variable explorer
df_y = pd.DataFrame(Y)

# for missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableEncoderX = LabelEncoder()
X[:,0] = lableEncoderX.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features= [0])
X =  oneHotEncoder.fit_transform(X).toarray()
lableEncoderY = LabelEncoder()
Y = lableEncoderX.fit_transform(Y)

# Spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(X,Y,test_size = 0.2,random_state = 0)