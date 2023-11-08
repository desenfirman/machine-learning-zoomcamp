# DataTalksClub Machine Learning Zoomcamp Midterm Project: Predict FIFA 20 Player with Trait Early Crosser

## Problem Description

In FIFA 20, some of player have traits which represent the in-game characteristic that give the player a unique skills & distinct player style between the players. There are so many player traits that available in the FIFA 20, for example: Leadership, Clinical Finisher, Speedster, Long Shooter and many more. In this case, the goal is to predict if a player has a trait an Early Crosser. An Early Crosser itself described as a player with a tendency to perform a direct cross to the open spaces of opponent's penalty area before reaching the end line or corner flag. This kind of player usually has a vision of the other teammates movement to the open-spaces in the front of the opponent's penalty area combined with a good long ball cross. While at the same time, there are multiple attributes that made the player has a Early Crosser trait.

The dataset that will be used to perform this following task is come from this url: https://www.kaggle.com/datasets/stefanoleone992/fifa-20-complete-player-dataset?select=players_20.csv

The dataset basically consists a FIFA player id and player name with the skill attribute (eg: attacking, shooting, physic, and more) which represented in numerical attributes, while it's also has a categorical attribute like team name, position, nationality, prefered_foot and many more. We will use some of this attribute to perform binary-classification task that produce a decision if the player has a trait Early Crosser or not.

## Personal Rationale

To make the complexity of model is small, the training will only use a attribute that related to the player's skills statistic (eg: dribbling, defending, attacking, etc). The player overall attributes in the specific position (eg: lb, lcb, cb) will be discarded. Some of the attributes also doesn't make sense to be used in the training phase. Eg: real_face which represent if the player has a real face in the FIFA games, player jersey number which doesn't seems has a impact to the training phase. Therefore, it will be discarded from the model building.

From the personal feature selection, here is the attribute that will be included into ML model building. The `sofifa_id` and `short_name` only act as identifier. The early crosser trait available in the column `player_traits`

Identifier
- sofifa_id
- short_name

Feature
- age
- height_cm
- weight_kg
- nationality
- club
- overall (overall score in FIFA 20)
- potential (the potential score that predicted int FIFA 20)
- value_eur
- wage_eur
- preferred_foot
- international_reputation
- weak_foot
- skill_moves
- work_rate
- release_clause_eur
- team_position
- nation_position
- pace
- shooting
- passing
- dribbling
- defending
- physic
- attacking_crossing
- attacking_finishing
- attacking_heading_accuracy
- attacking_short_passing
- attacking_volleys
- skill_dribbling
- skill_curve
- skill_fk_accuracy
- skill_long_passing
- skill_ball_control
- movement_acceleration
- movement_sprint_speed
- movement_agility
- movement_reactions
- movement_balance
- power_shot_power
- power_jumping
- power_stamina
- power_strength
- power_long_shots
- mentality_aggression
- mentality_interceptions
- mentality_positioning
- mentality_vision
- mentality_penalties
- mentality_composure
- defending_marking
- defending_standing_tackle
- defending_sliding_tackle

Target
- player_traits (only select the 'Early Crosser' and mark player which has the related trait using value=1 and value=0 respectively)


## Exploratory Data Analysis (EDA)

Here is the EDA that performed before the model building
1. Missing Value Checking: There are 9 columns of that has a missing value. Seven columns are classified as a numerical feature while the other is a categorical. The numerical features columns are release_clause_eur, pace, shooting, passing, dribbling, defending and physic and it's safe to fill it with zero value, since the zero itself represent the lowest possible value of those features. While the categorical features like team_position and nation_position will be filled with 'OTHER' category

2. Imbalance Class Check: The data ratio between a player with trait Early Crosser and Not is 10:90. Therefore, it's necessary to make the data distribution ratio is same for both class to avoid the higher True Negative prediction in the model. In this case, the RandomUnderSample strategies is used to perform a sampling of the Negative class, so the distribution will be same with the Positive class.

3. Checking the player distribution. eg: Average, STD dev, Min-Max value and the Quartile value for the each features.

4. Perform a feature importance analysis towards the target class. The categorical features used mutual info_score, while the numerical features used correlation score as a method to peform feature importance analysis. During the feature importance analysis, only one feature that didn't have any correlation towards the target, which is `preferred_foot`. Therefore, it will be excluded from model building phase

During the feature importance analysis, there are interesting fact that the feature importance analysis suprisingly showed that the highly correlated attribute of early-crosser player is an attribute that related to the **movement** (agility, acceleration, sprint_speed, pace) rather than the attribute that related to the crossing itself (eg: attacking_crossing, passing, skill_curve, mentality_vision).


## Model Training

There are three models were trained on the dataset. The dataset were splitted as train/val/test by ratio of 60/20/20 %. Here is the summary of training and the ROC metric that scored from the validation datasets:

| Model                                                 | Model Param (The best model if tuned)                | ROC AUC Score on val dataset (in %) |
|-------------------------------------------------------|---------------------------------|-------------------------------|
| Logistic Regression                                   | `LogisticRegression(random_state=42, solver='liblinear')`                                                       | 75.275                         |
| Logistic Regression (tuned with RandomizedCVSearch)   | `LogisticRegression(C=0.5, max_iter=1000, random_state=42, solver='liblinear', tol=1e-05)`                         | 91.672                         |
| XG Boost (tuned with RandomizedCVSearch)              | `XGBClassifier(min_child_weight=10, eta=0.1, max_depth=3)`                                                       | 91.721                         |

The Best model is using XGBClassifier. More details on: `notebook.ipynb`

