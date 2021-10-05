
import pandas as pd
import random
from time import time
import numpy as np

# those are optional and are not necessary for training
random.seed(73)


def split_train_test(x, y, test_ratio=0.1):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    # delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]


def heart_disease_data():
    data = pd.read_csv("./data/framingham.csv")
    # drop rows with missing values
    data = data.dropna()
    # drop some features
    data = data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])
    # balance data
    grouped = data.groupby('TenYearCHD')
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    print(data['TenYearCHD'].value_counts())
    # extract labels
    # y = torch.tensor(data["TenYearCHD"].values).float().unsqueeze(1)
    y = np.array(data["TenYearCHD"].values, dtype=float)[:,None, None]
    # y = y[:, None]
    data = data.drop(columns=["TenYearCHD"])
    # standardize data
    data = (data - data.mean()) / data.std()
    x = np.array(data.values, dtype=float)[:,None]
    return split_train_test(x, y)

def titanic_data():
    train_data = pd.read_csv('./data/titanic/train.csv')

    train_data = train_data.dropna()

    train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
    train_data.drop('SibSp', axis=1, inplace=True)
    train_data.drop('Parch', axis=1, inplace=True)
    train_data.drop('Cabin', axis=1, inplace=True)
 

    training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
    training.drop('Sex_female', axis=1, inplace=True)
    training.drop('PassengerId', axis=1, inplace=True)
    training.drop('Name', axis=1, inplace=True)
    training.drop('Ticket', axis=1, inplace=True)
    print(training['Survived'].value_counts())


    y_train = np.array(training['Survived'].values, dtype=float)[:,None, None]
    train = training.drop(columns=['Survived'])
    train = (train - train.mean()) / train.std()
    x_train = np.array(train.values, dtype=float)[:,None]

    return split_train_test(x_train, y_train)

def random_data(m=1024, n=2):
    # data separable by the line `y = x`
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)
    y_train = (x_train[:, 0] >= x_train[:, 1]).float().unsqueeze(0).t()
    y_test = (x_test[:, 0] >= x_test[:, 1]).float().unsqueeze(0).t()
    return x_train, y_train, x_test, y_test

