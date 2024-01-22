import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

import pickle

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

dataset_file_path = "./data/labeled_dataset.csv"

non_used_cols = [
    'track_id',
    'album_name',
    'track_name',
    'time_signature'
]

target_column = 'is_liked'
logging.info("Loading dataset")
df = pd.read_csv(dataset_file_path)

logging.info(f"Setting column `{target_column}` as target column")
df['target'] = df['is_liked']
df = df.drop(columns=['is_liked'])

logging.info("Removing records that has n/a value ...")
df = df[~df.isnull().any(axis=1)]
df

logging.info("Performing imbalance class handling by oversampling it (using SMOTENC) and downsampling it using RandomUnderSampler")
X_res, y_res = (df.drop(columns=['target'] + non_used_cols), df['target'])

over = SMOTENC(categorical_features=['artists', 'track_genre'], sampling_strategy=0.05, random_state=42)
# # fit and apply the transform
X_res, y_res = over.fit_resample(X_res, y_res)
# define undersampling strategy
under = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_res, y_res = under.fit_resample(X_res, y_res)
df = pd.concat([X_res, y_res], axis=1)

logging.info(f"Non used feature for this training session is: {non_used_cols}")

logging.info("Preparing the ready to use dataset")
used_cols = [
    col for col in df.drop(columns=['target']).columns
    if col not in non_used_cols
]
used_cols = used_cols + ['target']
df = df[used_cols]

logging.info("Performing train test split with ratio of 60/20/20")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values

del df_train['target']
del df_val['target']
del df_test['target']

logging.info("Converting train/test/val feature into dictvectorizer")
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

logging.info("Training LightGBMClassifier Model")
model = LGBMClassifier(
  boosting_type='gbdt',
  learning_rate=0.5,
  min_child_weight=1,
  n_estimators=100,
  num_leaves=20,
  random_state=42
)
model.fit(X_train, y_train)
logging.info("Succesfully trained model")

y_pred = model.predict_proba(X_val)[:, 1]
logging.info(f"Score on validation set: {roc_auc_score(y_val, y_pred)}")

y_pred = model.predict_proba(X_test)[:, 1]
logging.info(f"Score on test set: {roc_auc_score(y_test, y_pred)}")

output_model_file = './model/final_model.bin'
output_dv_file = './model/final_dv.bin'

logging.info(f"Saving model into {output_model_file}")
with open(output_model_file, 'wb') as f_out: 
    pickle.dump(model, f_out)

logging.info(f"Saving model into {output_dv_file}")
with open(output_dv_file, 'wb') as f_out: 
    pickle.dump(dv, f_out)
