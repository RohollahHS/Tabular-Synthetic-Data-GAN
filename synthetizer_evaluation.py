import pandas as pd
import sklearn as sk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn import (
    metrics,
)  # Import scikit-learn metrics module for accuracy calculation


def read_preprocess_data(file_name="tickets"):
    df = pd.read_csv(f"data/{file_name}.csv")

    date_columns = ["creation_date", "view_date", "action_date"]

    for c in date_columns:
        df[c] = df[c].apply(lambda x: x[:16])

    df[date_columns] = df[date_columns].apply(
        pd.to_datetime, format="%Y-%m-%d %H:%M:%S"
    )

    df["view_creation_distance"] = df["view_date"] - df["creation_date"]
    df["action_view_distance"] = df["action_date"] - df["view_date"]

    df["view_creation_distance_minute"] = (
        df["view_creation_distance"].dt.total_seconds().div(60).astype(int)
    )

    df["action_view_distance_minute"] = (
        df["action_view_distance"].dt.total_seconds().div(60).astype(int)
    )

    df["delta_creation_date"] = (
        df["creation_date"]
        .sort_values()
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
        .astype(int)
    )

    drop_columns = [
        "creation_date",
        "view_date",
        "action_date",
        "view_creation_distance",
        "action_view_distance",
    ]

    return df.drop(columns=drop_columns)


def decesion_tree(X_train, y_train, X_syn, y_syn, X_test, y_test, max_depth=3):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    original_data_acc = metrics.accuracy_score(y_test, y_pred)

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(X_syn, y_syn)
    y_pred = clf.predict(X_test)
    synthetic_data_acc = metrics.accuracy_score(y_test, y_pred)

    print('\nDecesion Tree:')
    print(f"Original data accuracy on test set: {original_data_acc}")
    print(f"Synthetic data accuracy on test set: {synthetic_data_acc}")


def SVM(X_train, y_train, X_syn, y_syn, X_test_, y_test, kernel='linear', max_iter=500):
    continous_columns = ['view_creation_distance_minute', 'action_view_distance_minute', 'delta_creation_date']

    X_test = deepcopy(X_test_)

    try:
        for col in continous_columns:
            scaler = MinMaxScaler()
            scaler.fit(X_train[col].values.reshape(-1, 1))
            X_train[col] = scaler.transform(X_train[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    except:
        pass

    clf = svm.SVC(kernel=kernel, max_iter=max_iter)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    original_data_acc = metrics.accuracy_score(y_test, y_pred)

    X_test = deepcopy(X_test_)
    try:
        for col in continous_columns:
            scaler = MinMaxScaler()
            scaler.fit(X_syn[col].values.reshape(-1, 1))
            X_syn[col] = scaler.transform(X_syn[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    except:
        pass

    clf = svm.SVC(kernel=kernel, max_iter=max_iter)
    clf = clf.fit(X_syn, y_syn)
    y_pred = clf.predict(X_test)
    synthetic_data_acc = metrics.accuracy_score(y_test, y_pred)

    print('\nSVM:')
    print(f"Original data accuracy on test set: {original_data_acc}")
    print(f"Synthetic data accuracy on test set: {synthetic_data_acc}")


def MLP(X_train, y_train, X_syn, y_syn, X_test_, y_test,
        optimizer='adam', batch_size=100, max_iter=100):
    continous_columns = ['view_creation_distance_minute', 'action_view_distance_minute', 'delta_creation_date']

    X_test = deepcopy(X_test_)

    try:
        for col in continous_columns:
            scaler = MinMaxScaler()
            scaler.fit(X_train[col].values.reshape(-1, 1))
            X_train[col] = scaler.transform(X_train[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    except:
        pass

    clf = MLPClassifier(
        solver=optimizer,
        batch_size=batch_size,
        max_iter=max_iter,
    )
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    original_data_acc = metrics.accuracy_score(y_test, y_pred)

    X_test = deepcopy(X_test_)
    try:
        for col in continous_columns:
            scaler = MinMaxScaler()
            scaler.fit(X_syn[col].values.reshape(-1, 1))
            X_syn[col] = scaler.transform(X_syn[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    except:
        pass

    clf = MLPClassifier(
        solver=optimizer,
        batch_size=batch_size,
        max_iter=max_iter,
    )
    clf = clf.fit(X_syn, y_syn)
    y_pred = clf.predict(X_test)
    synthetic_data_acc = metrics.accuracy_score(y_test, y_pred)

    print('\nMLP:')
    print(f"Original data accuracy on test set: {original_data_acc}")
    print(f"Synthetic data accuracy on test set: {synthetic_data_acc}")



def synthetic_evaluation(
    original_data,
    synthetic_data,
    drop_columns=[],
    label_column="customer_problem_resolved",
    test_size=0.3,
    max_depth_tree=10,
    kernel_svm='linear'
):
    for col in drop_columns:
        original_data.drop(col, inplace=True, axis=1)
        synthetic_data.drop(col, inplace=True, axis=1)

    categorical_columns = []
    for col, dtype in zip(original_data.columns, original_data.dtypes):
        if dtype == "O":
            categorical_columns.append(col)

    for col in categorical_columns:
        onehot = pd.get_dummies(original_data[col])
        original_data.drop(col, inplace=True, axis=1)
        original_data = original_data.join(onehot)

        onehot = pd.get_dummies(synthetic_data[col])
        synthetic_data.drop(col, inplace=True, axis=1)
        synthetic_data = synthetic_data.join(onehot)

    idx = original_data[original_data[label_column] == True].index
    original_data.loc[idx, label_column] = 1
    idx = original_data[original_data[label_column] == False].index
    original_data.loc[idx, label_column] = 0

    idx = synthetic_data[synthetic_data[label_column] == True].index
    synthetic_data.loc[idx, label_column] = 1
    idx = synthetic_data[synthetic_data[label_column] == False].index
    synthetic_data.loc[idx, label_column] = 0


    features = [col for col in original_data.columns if col != label_column]
    X_original = original_data[features]
    y_original = original_data[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y_original, test_size=test_size, random_state=1
    )
    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)

    X_synthetic = synthetic_data[features]
    y_synthetic = synthetic_data[label_column]
    y_synthetic = y_synthetic.values.astype(int)

    decesion_tree(X_train, y_train, X_synthetic, y_synthetic, X_test, y_test, max_depth=max_depth_tree)
    SVM(X_train, y_train, X_synthetic, y_synthetic, X_test, y_test, kernel=kernel_svm)
    MLP(X_train, y_train, X_synthetic, y_synthetic, X_test, y_test)



original_data = read_preprocess_data("tickets")
synthetic_data = read_preprocess_data("fianl_synthetic_data")

# drop_columns = ['task_type', 'customer_satisfaction', 'user_actioned', 'user_team']
drop_columns = []

synthetic_evaluation(
    original_data,
    synthetic_data,
    drop_columns=drop_columns,
    label_column="customer_problem_resolved",
    test_size=0.3,
    max_depth_tree=3,
    kernel_svm='linear'
)

print('Finish')
