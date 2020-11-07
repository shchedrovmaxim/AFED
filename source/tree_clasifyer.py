import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt 
import sklearn.metrics as metrics
import pandas_profiling as pp
import os


data_train = pd.read_csv('data/train_classification_test_size_0.2.csv')
data_test = pd.read_csv('data/test_classification_test_size_0.2.csv')

y_train = data_train['INSTALLMENTS_PURCHASES']
X_train = data_train.drop('INSTALLMENTS_PURCHASES',axis=1)
y_test = data_test["INSTALLMENTS_PURCHASES"]
X_test = data_test.drop('INSTALLMENTS_PURCHASES', axis=1)

# profile = pp.ProfileReport(data_train, minimal=False, explorative=True)
# profile.to_file("report_raw.html")

def split_y(y):
    if y > 500:
        y = 2
    elif y < 100:
        y = 0
    else:
        y = 1
    return y
    
def calc_metrics(y_true, y_pred):
    return {
        "acc": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred,average=None),
        "precision": metrics.precision_score(y_true, y_pred,average=None),
        "f1": metrics.f1_score(y_true, y_pred,average=None)
    }

def show_metrics(y_true, y_proba, thresholds, data_type, balance_type, path):
    results = []
    return_results = []
    thresh_v = [val / 100 for val in range(50, 100, 1)]
    for threshold in thresh_v:
        y_pred = []
        # y_pred = (y_proba[:, 1] > threshold).astype(int)
        for y in y_proba:
            if y[np.argmax(y)] > threshold:
                y_pred.append(np.argmax(y))
            else:
                y_pred.append(2)
        metrics = calc_metrics(y_true, y_pred)
        metrics["threshold"] = threshold
        results.append(metrics)
        if threshold in thresholds:
            return_results.append(metrics)
    
    plt.figure(figsize=(16, 9))
    plt.title(f"Metrics by threshold. Dataset: {data_type}. Balancing: {balance_type}")
    plt.xlabel("threshold")
    plt.plot(thresh_v, [res["recall"] for res in results], label="Recall")
    plt.plot(thresh_v, [res["precision"] for res in results], label="Precision")
    plt.plot(thresh_v, [res["acc"] for res in results], label="Accuracy")
    plt.plot(thresh_v, [res["f1"] for res in results], label="F1 Score")
    plt.legend()
    plt.savefig(os.path.join(path, f"{data_type}_{balance_type}"))
    # plt.show()
    
    return return_results


y_train = y_train.apply(split_y)
y_test = y_test.apply(split_y)

columns = ["BALANCE","PURCHASES","ONEOFF_PURCHASES", "CASH_ADVANCE", "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "PAYMENTS"]
for column in columns:
    X_train[column] = np.log10(X_train[column]+10)
    X_test[column] = np.log10(X_test[column]+10)

# profile = pp.ProfileReport(X_train, minimal=False, explorative=True)
# profile.to_file("report_not_raw.html")


thresholds = [0.8, 0.85, 0.9, 0.95]
results = []
for weights in ["balanced", "unbalanced"]:
    reg = DecisionTreeClassifier(
        max_depth=6, min_samples_split=20,
        min_samples_leaf=5, random_state=50,
        class_weight=weights if weights == "balanced" else None
    ).fit(X_train, y_train)
    train_metrics = show_metrics(
        y_train, reg.predict_proba(X_train), thresholds,
        "train", weights, "results"
    )
    [metrics.update({"type": weights, "data": "train"}) for metrics in train_metrics]
    test_metrics = show_metrics(
        y_test, reg.predict_proba(X_test), thresholds,
        "test", weights, "results"
    )
    [metrics.update({"type": weights, "data": "test"}) for metrics in test_metrics]
    
    results += train_metrics + test_metrics

pd.DataFrame(results).to_csv("results/results.csv", index=False)



