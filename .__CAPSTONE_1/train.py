import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import pickle

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

dataset_file_path = "./data/gsod_jakarta_prepared.csv"
N_DAY_RANGE = (1, 8)

missing_value_filler = {
    'avg_temp': 9999.9,
    'avg_dew_point': 9999.9,
    'avg_sea_level_point': 9999.9,
    'avg_wind_speed': 999.9,
    'total_precipitation': 99.99,
    'flag_precipitation': 'OTHER'
}
non_used_cols = [
    'date',
    'year',
    'mo',
    'da',
    'avg_wind_speed_prev_1_day'
    'avg_dew_point_prev_1_day'
    'avg_sea_level_point_prev_1_day'
    'avg_temp_prev_1_day'
    'avg_temp_prev_3_day'
    'avg_dew_point_prev_3_day'
    'avg_sea_level_point_prev_3_day'
]

target_column = 'rain_drizzle'
logging.info("Loading dataset")
df = pd.read_csv(dataset_file_path)

logging.info(f"Setting column `{target_column}` as target column")
df['target'] = df['rain_drizzle']
df = df.drop(columns=['rain_drizzle'])

logging.info("Filling missing value...")
for col, value in missing_value_filler.items():
    df = df.fillna(value={
        f'{col}_prev_{i}_day': value
        for i in range(*N_DAY_RANGE)
    })
df = df.dropna()

logging.info("Performing imbalance class removal by RandomUnderSampler")
undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_res, y_res = undersample.fit_resample(df.drop(columns='target'), df['target'])
df = pd.concat([X_res, y_res], axis=1)

logging.info(f"Non used feature for this training session is: {non_used_cols}")

logging.info("Preparing the ready to use dataset")
used_cols = [
    col for col in df.drop(columns=['date', 'target']).columns
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

logging.info("Training RandomForestClassifier Model")
model = RandomForestClassifier(
  max_depth=10, max_leaf_nodes=50, max_samples=250,
  min_samples_leaf=10, min_samples_split=50,
  n_estimators=250, random_state=42
)
model.fit(X_train, y_train)
logging.info("Succesfully trained model")

y_pred = model.predict_proba(X_val)[:, 1]
logging.info(f"Score on validation set: {roc_auc_score(y_val, y_pred)}")

y_pred = model.predict_proba(X_test)[:, 1]
logging.info(f"Score on test set: {roc_auc_score(y_test, y_pred)}")

output_model_file = './model/final_model_.bin'
output_dv_file = './model/final_dv.bin'

logging.info(f"Saving model into {output_model_file}")
with open(output_model_file, 'wb') as f_out: 
    pickle.dump(model, f_out)

logging.info(f"Saving model into {output_dv_file}")
with open(output_dv_file, 'wb') as f_out: 
    pickle.dump(dv, f_out)
