# coding=utf8

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif


# Po wczytaniu app_events, łączymy rekordy odpowiadające tym samym wydarzeniom, zachowując unikalne app id
print("# Processing App Events")
app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
app_ev = app_ev.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))

#Następnie stworzone mapowanie zostaje wmergowane do eventów, używając jako indeksu event_id
print("# Processing Events")
events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
events["app_id"] = events["event_id"].map(app_ev)

#Odrzucamy wiersze z brakującymi wartościami, zależy nam tylko na rzeczywistych mapowaniach app_ids
events = events.dropna()
events = events[["device_id", "app_id"]]

# Odrzucenie duplikatów i reset indeksu
events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")

# Rozszerzamy do wiersza per app_id
events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']

#Wczytujemy db z informacją o modelach telefonów
print("# Processing Phone Brand")
pbd = pd.read_csv("../input/phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)



print("# Processing Train & Test set")

train = pd.read_csv("../input/gender_age_train.csv",
                    dtype={'device_id': np.str})
train.drop(["age", "gender"], axis=1, inplace=True)

Y = train["group"]
label_group = LabelEncoder()
Y = label_group.fit_transform(Y)


print('Merging dataframes')

Df = train
Df = pd.merge(Df, pbd, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model"] = Df["device_model"].apply(
    lambda x: "device_model:" + str(x))



print('Concatenating features')

f1 = Df[["device_id", "phone_brand"]]   # phone_brand
f2 = Df[["device_id", "device_model"]]  # device_model
f3 = events[["device_id", "app_id"]]    # app_id

f1.columns.values[1] = "feature"
f2.columns.values[1] = "feature"
f3.columns.values[1] = "feature"

FLS = pd.concat((f1, f2, f3), axis=0, ignore_index=True)

print('building user-item feature mapping')

device_ids = FLS["device_id"].unique()
feature_cs = FLS["feature"].unique()

data = np.ones(len(FLS))
dec = LabelEncoder().fit(FLS["device_id"])
row = dec.transform(FLS["device_id"])
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix(
    (data, (row, col)), shape=(len(device_ids), len(feature_cs)))

sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]

train_row = dec.transform(train["device_id"])
train_sp = sparse_matrix[train_row, :]

X_train, X_val, y_train, y_val = train_test_split(
    train_sp, Y, train_size=.90, random_state=10)

print("# Feature Selection - Bag of features")
selector = SelectPercentile(f_classif, percentile=20)

selector.fit(X_train, y_train)

X_train = selector.transform(X_train)
X_val = selector.transform(X_val)

train_sp = selector.transform(train_sp)

print("# Num of Features: ", X_train.shape[1])

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_val, y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gblinear",
    "max_depth": 6,
    "eval_metric": "mlogloss",
    "eta": 0.07,
    "silent": 1,
    "alpha": 3,
}

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 40, evals=watchlist,
                early_stopping_rounds=25, verbose_eval=True)
