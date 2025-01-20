### This file will combine 2 datsets into 1 from Kaggle
import pandas as pd

from config import DATASET1_PATH, DATASET2_PATH
from helpers import  to_snake_case, sleep_col_mediate, study_hours_col_mediate, depression_col_mediate, sleep_col_mediate_to_int, study_hours_col_mediate_to_int
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv(DATASET1_PATH)
df2 = pd.read_csv(DATASET2_PATH)

### Columns do not match or not identical in this case
print(df1.columns)
print(df2.columns)
print("---------------------------------")

### As df2 has much more columns and similar goals, we will subsume df2 under the state of df1

### First lets drop redundant columns
df2 = df2.drop(columns=['university', 'degree_level', 'degree_major',
       'academic_year', 'cgpa', 'residential_status', 'campus_discrimination',
       'sports_engagement', 'anxiety', 'isolation', 'future_insecurity',
       'social_relationships', 'stress_relief_activities'])

df1 = df1.drop(columns=['Family History of Mental Illness'])

df1.columns = [to_snake_case(col) for col in df1.columns]
df2.columns = [to_snake_case(col) for col in df2.columns]

df1 = df1.rename(columns={'have_you_ever_had_suicidal_thoughts_?':'suicidal_thoughts'})
df2 = df2.rename(columns={'academic_workload':'study_hours','financial_concerns':'financial_stress', 
                          'average_sleep':'sleep_duration'})

df2['dietary_habits'] = pd.Series(dtype='object')
df2['suicidal_thoughts'] = pd.Series(dtype='object')

df1['sleep_duration'] = df1['sleep_duration'].map(lambda name: sleep_col_mediate(name))

df1['study_hours'] = df1['study_hours'].map(lambda name: f'{name} hrs')
df2['study_hours'] = df2['study_hours'].map(lambda name: study_hours_col_mediate(name))

df1['depression'] = df1['depression'].map({'Yes': 1, 'No': 0})
df2['depression'] = df2['depression'].map(lambda name: depression_col_mediate(name))

print(df1.columns)
print(df2.columns)
print(df1.columns.size)
print(df2.columns.size)

### Combine 2 datset after mediating both of them
combined_dataset = pd.concat([df1, df2], axis=0, ignore_index=True)

### We have missing values for the second dataset. Moreover there is no matching values or columns that we could substitude
### We will perform predictions using KNN
dietary_habits_encoder = LabelEncoder()
suicidal_thoughts_encoder = LabelEncoder()

combined_dataset['dietary_habits_encoded'] = dietary_habits_encoder.fit_transform(combined_dataset['dietary_habits'].fillna('Missing'))
combined_dataset['dietary_habits_encoded'] = combined_dataset['dietary_habits_encoded'].where(combined_dataset['dietary_habits'].notnull(), None)

combined_dataset['suicidal_thoughts_encoded'] = suicidal_thoughts_encoder.fit_transform(combined_dataset['suicidal_thoughts'].fillna('Missing'))
combined_dataset['suicidal_thoughts_encoded'] = combined_dataset['suicidal_thoughts_encoded'].where(combined_dataset['suicidal_thoughts'].notnull(), None)

combined_dataset['sleep_duration_encoded'] = combined_dataset['sleep_duration'].map(lambda sleep: sleep_col_mediate_to_int(sleep))
combined_dataset['study_hours_encoded'] = combined_dataset['study_hours'].map(lambda study: study_hours_col_mediate_to_int(int(''.join(filter(str.isdigit, study)))))


### Using correlation matrics these are 3 top features that depend on each other
knn_data_dietary_habits = combined_dataset[['dietary_habits_encoded', 'financial_stress', 'depression', 'sleep_duration_encoded']]
knn_data_suicidal_thoughts = combined_dataset[['suicidal_thoughts_encoded', 'study_satisfaction', 'depression', 'study_hours_encoded']]

imputer_dietary_habits = KNNImputer(n_neighbors=1)
imputed_data_dietary_habits = imputer_dietary_habits.fit_transform(knn_data_dietary_habits)

imputer_suicidal_thoughts = KNNImputer(n_neighbors=5)
imputed_data_suicidal_thoughts = imputer_suicidal_thoughts.fit_transform(knn_data_suicidal_thoughts)

combined_dataset['dietary_habits_encoded'] = imputed_data_dietary_habits[:, 0].round().astype(int)
combined_dataset['dietary_habits'] = dietary_habits_encoder.inverse_transform(
    combined_dataset['dietary_habits_encoded'].astype(int))

combined_dataset['suicidal_thoughts_encoded'] = imputed_data_suicidal_thoughts[:, 0].round().astype(int)
combined_dataset['suicidal_thoughts'] = suicidal_thoughts_encoder.inverse_transform(
    combined_dataset['suicidal_thoughts_encoded'].astype(int))

combined_dataset = combined_dataset.drop(
    columns=['dietary_habits_encoded', 'sleep_duration_encoded', 'suicidal_thoughts_encoded', 'study_hours_encoded'])

### Now we will use IQR to get rid of too big or too small values
iqr_columns = combined_dataset.drop(columns=['gender', 'dietary_habits', 'suicidal_thoughts', 'depression'])
iqr_columns['sleep_duration'] = iqr_columns['sleep_duration'].map(lambda name: sleep_col_mediate_to_int(name))
iqr_columns['study_hours'] = iqr_columns['study_hours'].map(lambda name: study_hours_col_mediate_to_int(int(''.join(filter(str.isdigit, name)))))

Q1 = iqr_columns.quantile(0.25)
Q3 = iqr_columns.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask = ~((iqr_columns < lower_bound) | (iqr_columns > upper_bound)).any(axis=1)
cleaned_dataset = combined_dataset[mask]

cleaned_dataset.to_csv("../CleansedDataset.csv", index=False)

normalized_dataset = cleaned_dataset
normalized_dataset['gender'] = normalized_dataset['gender'].map({'Male' : 0, 'Female' : 1})
normalized_dataset['suicidal_thoughts'] = normalized_dataset['suicidal_thoughts'].map({'No' : 0, 'Yes' : 1})
normalized_dataset['sleep_duration'] = normalized_dataset['sleep_duration'].map(lambda name: sleep_col_mediate_to_int(name))
normalized_dataset['study_hours'] = normalized_dataset['study_hours'].map(lambda name: study_hours_col_mediate_to_int(int(''.join(filter(str.isdigit, name)))))

dietary_habits_encoded = pd.get_dummies(normalized_dataset['dietary_habits'], prefix='dietary_habits')
normalized_dataset = pd.concat([normalized_dataset, dietary_habits_encoded], axis=1)
normalized_dataset = normalized_dataset.drop(columns=['dietary_habits'])

normalized_dataset.to_csv("../NormalizedDataset.csv", index=False)
