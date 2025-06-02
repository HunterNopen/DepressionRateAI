import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy import stats
from config import BEST_MODEL_PATH, NORNALIZED_DATASET_PATH
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model

def predict_depression(suicidal_thoughts, academic_pressure, age, **kwargs):
    
    #loaded_model = load_model(BEST_MODEL_PATH)
    loaded_model = joblib.load("../rf_model.keras")
        
    scaler = joblib.load('../scaler.pkl')
    
    default_values = {
        'gender': 0,
        'study_satisfaction': 3,
        'sleep_duration': 7,
        'study_hours': 3,
        'financial_stress': 2,
        'dietary_habits': "Moderate"
    }
    
    for key, value in kwargs.items():
        if key in default_values:
            default_values[key] = value
    
    input_data = {
        'suicidal_thoughts': suicidal_thoughts,
        'academic_pressure': academic_pressure,
        'age': age,
        **default_values
    }
    
    dietary_habits_categories = ['Healthy', 'Moderate', 'Unhealthy']
    dietary_habits_encoded = pd.get_dummies(
        [input_data['dietary_habits']],
        prefix='dietary_habits'
    )
    for category in dietary_habits_categories:
        col_name = f'dietary_habits_{category}'
        if col_name not in dietary_habits_encoded:
            dietary_habits_encoded[col_name] = 0

    input_data.pop('dietary_habits')
    input_data.update(dietary_habits_encoded.iloc[0].to_dict())

    feature_order = [
        'gender', 'age', 'academic_pressure', 'study_satisfaction', 'sleep_duration',
        'suicidal_thoughts', 'study_hours', 'financial_stress',
        'dietary_habits_Healthy', 'dietary_habits_Moderate','dietary_habits_Unhealthy'
    ]
    input_ordered = {key: input_data[key] for key in feature_order}

    input_df = pd.DataFrame([input_ordered])
    
    #input_scaled = scaler.transform(input_df)

    prediction = loaded_model.predict_proba(input_df).flatten()
    result = f"Depression: Yes - {(prediction[1]*100):.2f}%, No - {(prediction[0]*100):.2f}%"

    return result

def standardize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    standarized_data = scaler.fit_transform(data)
    
    data_frame = pd.DataFrame(data = standarized_data, columns = data.columns)
    
    return data_frame

def correlation_matrix(data):
    correlation_matrix = data.corr()

    plt.imshow(correlation_matrix, cmap='Blues')
    plt.colorbar()
    
    variables = []
    for i in correlation_matrix.columns:
        variables.append(i)
    
    plt.xticks(range(len(correlation_matrix)), variables, rotation=45, ha='right')
    plt.yticks(range(len(correlation_matrix)), variables)
    
    plt.show()
    

if __name__ == "__main__":
    
    data = pd.read_csv(NORNALIZED_DATASET_PATH)
    
    standardized_data = standardize_data(data)
    
    correlation_matrix(standardized_data)
    
    data['gender'] = data['gender'].map({ 0 : "Male", 1 : "Female"})
    sns.catplot(kind='bar', x = "gender", y = "depression", data = data, hue="gender")
    
    depression_males = data[data["gender"] == "Male"]["depression"]
    depression_females = data[data["gender"] == "Female"]["depression"]
    t_gender, p_gender = stats.ttest_ind(depression_males, depression_females)
    print("---------------------------------------------------")
    print(f"t_stats for Depression vs Gender : {t_gender:.2f}")
    print(f"p_stats for Depression vs Gender : {p_gender:.2f}")
    print("---------------------------------------------------")
    
    sns.catplot(kind='bar', x = "academic_pressure", y = "depression", data = data, hue="academic_pressure")
    academic_pressure_levels = sorted(data["academic_pressure"].unique()) 
    for level in academic_pressure_levels:
        group1 = data[data["academic_pressure"] == level]["depression"]
        group2 = data[data["academic_pressure"] != level]["depression"] 
        t_ap, p_ap = stats.ttest_ind(group1, group2)
        print(f"Depression vs Academic Pressure Level {level} : t_stats - {t_ap:.2f} | p_stats = {p_ap:.2f}")
        print("~~~")
    print("---------------------------------------------------")
    
    data['suicidal_thoughts'] = data['suicidal_thoughts'].map({ 0 : "No", 1 : "Yes"})
    sns.catplot(kind='bar', x = "suicidal_thoughts", y = "depression", data = data, hue="suicidal_thoughts", order = ['No', 'Yes'])
    
    suicidal_thoughts_positive = data[data["suicidal_thoughts"] == "Yes"]["depression"]
    suicidal_thoughts_negative = data[data["suicidal_thoughts"] == "No"]["depression"]
    t_st, p_st = stats.ttest_ind(suicidal_thoughts_positive, suicidal_thoughts_negative)
    print(f"t_stats for Depression vs Suicidal Thought : {t_st:.2f}")
    print(f"p_stats for Depression vs Suicidal Thought : {p_st:.2f}")
    print("---------------------------------------------------")
    
    sns.catplot(kind='bar', x = "financial_stress", y = "depression", data = data, hue="financial_stress")
    
    X = standardized_data.drop(columns=["depression"])
    y = standardized_data["depression"]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    sns.catplot(kind='bar', x="Importance", y="Feature", data=feature_importance)
    plt.title('Feature Importance')
    plt.show()
    
    sns.countplot(x='depression', data=data)
    plt.title('Distribution of Depression')
    plt.show()
    
    ### Roman - Ja
    prediction = predict_depression(suicidal_thoughts = 0, academic_pressure = 5, age = 19, 
            gender = 0, study_satisfaction = 5, study_hours = 6, sleep_duration = 4, financial_stress = 1, dietary_habits = "Healthy")
    print(f"Roman Prediction: {prediction}")
    
    ### Saveliy
    prediction = predict_depression(suicidal_thoughts = 0, academic_pressure = 5, age = 20, 
            gender = 0, study_satisfaction = 2, study_hours = 3, sleep_duration = 5, financial_stress = 4, dietary_habits = "Moderate")
    print(f"Saveliy Prediction: {prediction}")
    
    ### Magda
    prediction = predict_depression(suicidal_thoughts = 1, academic_pressure = 2, age = 21, 
            gender = 1, study_satisfaction = 2, study_hours = 3, sleep_duration = 8, financial_stress = 3, dietary_habits = "Moderate")
    print(f"Magda Prediction: {prediction}")
    
    ### Monika
    prediction = predict_depression(suicidal_thoughts = 0, academic_pressure = 4, age = 19, 
            gender = 1, study_satisfaction = 4, study_hours = 5, sleep_duration = 8, financial_stress = 1, dietary_habits = "Healthy")
    print(f"Monika Prediction: {prediction}")
    
    ### Jakub
    prediction = predict_depression(suicidal_thoughts = 0, academic_pressure = 3, age = 22, 
            gender = 0, study_satisfaction = 1, study_hours = 6, sleep_duration = 6, financial_stress = 2, dietary_habits = "Unhealthy")
    print(f"Jakub Prediction: {prediction}")
    
    ### Tomasz1
    prediction = predict_depression(suicidal_thoughts = 0, academic_pressure = 1, age = 18, 
            gender = 1, study_satisfaction = 5, study_hours = 4, sleep_duration = 7, financial_stress = 1, dietary_habits = "Healthy")
    print(f"Tomasz1 Prediction: {prediction}")
    
    ### Tomasz2
    prediction = predict_depression(suicidal_thoughts = 1, academic_pressure = 5, age = 20, 
            gender = 0, study_satisfaction = 1, study_hours = 1, sleep_duration = 4, financial_stress = 5, dietary_habits = "Unhealthy")
    print(f"Tomasz2 Prediction: {prediction}")
    
    ### Maria
    prediction = predict_depression(suicidal_thoughts = 0, academic_pressure = 2, age = 21, 
            gender = 1, study_satisfaction = 3, study_hours = 5, sleep_duration = 8, financial_stress = 3, dietary_habits = "Moderate")
    print(f"Maria Prediction: {prediction}")
    
    ### Jakub2
    prediction = predict_depression(suicidal_thoughts = 0, academic_pressure = 4, age = 19, 
            gender = 0, study_satisfaction = 2, study_hours = 7, sleep_duration = 5, financial_stress = 4, dietary_habits = "Moderate")
    print(f"Jakub2 Prediction: {prediction}") 
    
    
