import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import pickle

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

logging.info("Loading dataset")
df = pd.read_csv("./data/players_20.csv")

loaded_column = [
    'sofifa_id',
    'short_name',
    'age',
    'height_cm',
    'weight_kg',
    'nationality',
    'club',
    'overall',
    'potential',
    'value_eur',
    'wage_eur',
    'preferred_foot',
    'international_reputation',
    'weak_foot',
    'skill_moves',
    'work_rate',
    'release_clause_eur',
    'team_position',
    'nation_position',
    'pace',
    'shooting',
    'passing',
    'dribbling',
    'defending',
    'physic',
    'attacking_crossing',
    'attacking_finishing',
    'attacking_heading_accuracy',
    'attacking_short_passing',
    'attacking_volleys',
    'skill_dribbling',
    'skill_curve',
    'skill_fk_accuracy',
    'skill_long_passing',
    'skill_ball_control',
    'movement_acceleration',
    'movement_sprint_speed',
    'movement_agility',
    'movement_reactions',
    'movement_balance',
    'power_shot_power',
    'power_jumping',
    'power_stamina',
    'power_strength',
    'power_long_shots',
    'mentality_aggression',
    'mentality_interceptions',
    'mentality_positioning',
    'mentality_vision',
    'mentality_penalties',
    'mentality_composure',
    'defending_marking',
    'defending_standing_tackle',
    'defending_sliding_tackle',
    'player_traits'
]

df = df[loaded_column].copy()
df['target'] = df['player_traits'].map(lambda x: 1 if 'Early Crosser' in str(x) else 0)
df = df.drop(columns=['player_traits'])
logging.info("Filling missing value...")
df = df.fillna(value={
    'release_clause_eur': 0,
    'team_position': 'OTHER',
    'nation_position': 'OTHER',
    'pace': 0,
    'shooting': 0,
    'passing': 0,
    'dribbling': 0,
    'defending': 0,
    'physic': 0
})
logging.info("Performing imbalance class removal by RandomUnderSampler")
undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_res, y_res = undersample.fit_resample(df.drop(columns='target'), df['target'])
df = pd.concat([X_res, y_res], axis=1)

logging.info("Identifying the dataset's categorical and numerical features")
categorical_cols, numerical_cols, non_used_cols = [], [], ['preferred_foot']
logging.info(f"Non used feature for this training session is: {non_used_cols}")

for i, col_name in enumerate(df.drop(columns=['sofifa_id', 'short_name', 'target']).columns):
    # Check if the column is numerical
    if pd.api.types.is_numeric_dtype(df[col_name]):
        numerical_cols.append(col_name)
    else:
        categorical_cols.append(col_name)

logging.info("Preparing the ready to use dataset")
df = df.copy()
used_cols = [col for col in categorical_cols if col != 'preferred_foot'] + numerical_cols + ['target']

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

logging.info("Training XGBClassifier Model")
model = XGBClassifier(
    objective='binary:logistic',
    nthread=8, seed=1, verbosity=1,
    min_child_weight=10,
    eta=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
logging.info("Succesfully trained model")

y_pred = model.predict_proba(X_val)[:, 1]
logging.info(f"Score on validation set: {roc_auc_score(y_val, y_pred)}")

y_pred = model.predict_proba(X_test)[:, 1]
logging.info(f"Score on test set: {roc_auc_score(y_test, y_pred)}")

output_model_file = './model/final_model_xgb.bin'
output_dv_file = './model/final_dv.bin'

logging.info(f"Saving model into {output_model_file}")
with open(output_model_file, 'wb') as f_out: 
    pickle.dump(model, f_out)

logging.info(f"Saving model into {output_dv_file}")
with open(output_dv_file, 'wb') as f_out: 
    pickle.dump(dv, f_out)
