import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/train.csv')
data = data.drop(["PassengerId", "Name", "Ticket"], axis=1)
y = data["Survived"]
data = data.drop(["Survived"], axis=1)

data["Age"] = data["Age"].fillna(28)
data["Embarked"] = data["Embarked"].fillna('S')

data["Cabin"].isna().sum(), len(data)
data = data.drop(["Cabin"], axis=1)
assert not data.isnull().values.any()

data["Sex"] = data["Sex"].astype('category')
data["Sex"] = data["Sex"].cat.codes

data = pd.get_dummies(data, columns=["Embarked"])

train_data, val_data, train_y, val_y = train_test_split(data, y, test_size=0.3, random_state=0)  # 0.3 - 30% идет на тест
lr = LogisticRegression()
#%%
lr.fit(train_data, train_y)
predicted = lr.predict(val_data)

print(accuracy_score(predicted, val_y))
test_data = pd.read_csv("data/test.csv")
test_data = test_data.drop(["PassengerId", "Name", "Ticket"], axis=1)
test_data["Age"] = test_data["Age"].fillna(28)
test_data["Embarked"] = test_data["Embarked"].fillna('S')
test_data = test_data.drop(["Cabin"], axis=1)
test_data["Sx"] = test_data["Sex"].astype('category')
test_data["Sex"] = test_data["Sex"].cat.codes
test_data = pd.get_dummies(test_data, columns=["Embarked"])

test_data["Fare"] = test_data["Fare"].fillna(train_data["Fare"].median())