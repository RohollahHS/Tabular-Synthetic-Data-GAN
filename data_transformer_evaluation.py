import pandas as pd
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn import metrics
from tabulate import tabulate



def read_preprocess_data(file_name, dir_path):
    df = pd.read_csv(f"{dir_path}/{file_name}.csv")

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

    return original_data_acc, synthetic_data_acc


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
            X_syn.loc[X_syn.index, col] = scaler.transform(X_syn[col].values.reshape(-1, 1))
            X_test.loc[X_test.index, col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    except:
        pass

    clf = svm.SVC(kernel=kernel, max_iter=max_iter)
    clf = clf.fit(X_syn, y_syn)
    y_pred = clf.predict(X_test)
    synthetic_data_acc = metrics.accuracy_score(y_test, y_pred)

    print('\nSVM:')
    print(f"Original data accuracy on test set: {original_data_acc}")
    print(f"Synthetic data accuracy on test set: {synthetic_data_acc}")

    return original_data_acc, synthetic_data_acc


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
            X_syn.loc[X_syn.index, col] = scaler.transform(X_syn[col].values.reshape(-1, 1))
            X_test.loc[X_test.index, col] = scaler.transform(X_test[col].values.reshape(-1, 1))
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

    return original_data_acc, synthetic_data_acc



def synthetic_evaluation(
    original_data,
    synthetic_data,
    drop_columns=[],
    label_column="customer_problem_resolved",
    test_size=0.3,
    max_depth_tree=10,
    kernel_svm='linear',
    max_iter_mlp=100,
    dir_path='data',
    model_name=None
):
    # for col in drop_columns:
    #     original_data.drop(col, inplace=True, axis=1)
    #     synthetic_data.drop(col, inplace=True, axis=1)

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

    origin_acc_dt,  syn_acc_dt  = decesion_tree(X_train, y_train, X_synthetic, y_synthetic, 
                                                X_test, y_test, max_depth=max_depth_tree)
    origin_acc_svm, syn_acc_svm = SVM(X_train, y_train, X_synthetic, y_synthetic, 
                                      X_test, y_test, kernel=kernel_svm)
    origin_acc_mlp, syn_acc_mlp = MLP(X_train, y_train, X_synthetic, y_synthetic, 
                                      X_test, y_test, max_iter=max_iter_mlp)
    
    return origin_acc_dt, syn_acc_dt, origin_acc_svm, syn_acc_svm, origin_acc_mlp, syn_acc_mlp


def start_evaluation(file_name, dir_path, model_name, drop_column=False):
    original_data = read_preprocess_data(file_name, dir_path)
    synthetic_data = read_preprocess_data(file_name, dir_path)

    if drop_column:
        drop_columns = ['task_type', 'customer_satisfaction', 'user_actioned', 'user_team']        
        
        for i in range(len(drop_columns)+1):
            original_data_copy = original_data.drop(columns=drop_columns[:i], axis=1)
            synthetic_data_copy = synthetic_data.drop(columns=drop_columns[:i], axis=1)

            (origin_acc_dt, 
             syn_acc_dt, 
             origin_acc_svm, 
             syn_acc_svm, 
             origin_acc_mlp, 
             syn_acc_mlp) = synthetic_evaluation(original_data_copy,
                                                 synthetic_data_copy,
                                                 drop_columns=drop_columns,
                                                 label_column="customer_problem_resolved",
                                                 test_size=0.3,
                                                 max_depth_tree=3,
                                                 kernel_svm='linear',
                                                 max_iter_mlp = 10,
                                                 dir_path=dir_path,
                                                 model_name=model_name)
        
            accs = [['DT', origin_acc_dt, syn_acc_dt],
                    ['SVM', origin_acc_svm, syn_acc_svm],
                    ['MLP', origin_acc_mlp, syn_acc_mlp]]

            print(f'Dropped Columns: {drop_columns[:i]}')
            print(tabulate(accs, headers=[f"{model_name}", 
                                        "Original Data", 
                                        "Synthetic Data"]))

            acc_records = open(f'{dir_path}/accuracy_records.txt', 'a')
            acc_records.write(f'Dropped Columns: {drop_columns[:i]}\n')
            acc_records.write(tabulate(accs, headers=[f"{model_name}", 
                                            "Original Data", 
                                            "Synthetic Data"]))
            acc_records.write('\n')
            acc_records.write(78*'_')
            acc_records.write('\n\n')
            
            acc_records.close()
    
    else:
        synthetic_evaluation(
            original_data,
            synthetic_data,
            drop_columns=drop_columns,
            label_column="customer_problem_resolved",
            test_size=0.3,
            max_depth_tree=3,
            kernel_svm='linear',
            max_iter_mlp = 10,
            dir_path=dir_path,
            model_name=model_name)
