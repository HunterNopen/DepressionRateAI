import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from config import NORNALIZED_DATASET_PATH

def knn_train(file_path, n_neighbors):
    
    data = pd.read_csv(file_path)

    ### Depression is the last column - our label
    X = data.drop(columns=['depression'])
    y = data['depression']

    ### Splitting into train and test sets
    print("KNN: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ### Training a KNN classifier
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)

    ### Evaluating the model
    y_pred = knn_model.predict(X_test)
    
    knn_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {knn_accuracy:.2f}")
    
    knn_report = classification_report(y_test, y_pred, target_names=['No Depression', 'Depression'])
    print("\nClassification Report:")
    print(knn_report)
    
    return knn_model, knn_report

def random_forest_train(file_path):
    
    data = pd.read_csv(file_path)
    
    ### Depression is the last column - our label
    X = data.drop(columns=['depression'])
    y = data['depression']
    
    ### Splitting into train and test sets
    print("RandomForest: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ### Training a RandomForest classifier
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    ### Evaluating the model
    rf_predictions = rf_model.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Accuracy: {rf_accuracy:.2f}")
  
    rf_classification_report = classification_report(y_test, rf_predictions, target_names=['No Depression', 'Depression'])
    print("\nClassification Report:")  
    print(rf_classification_report)
    
    return rf_model, rf_classification_report

def cnn_train(file_path):
    
    data = pd.read_csv(file_path)

    ### Depression is the last column - our label
    X = data.drop(columns=['depression'])
    y = data['depression']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_categorical = to_categorical(y)

    ### Splitting into train and test sets
    print("CNN: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)
    
    cnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy', metrics=['accuracy'])

    ### Training a KNN classifier
    cnn_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

    ### Evaluating the model
    cnn_predictions = cnn_model.predict(X_test)
    
    cnn_predicted_classes = np.argmax(cnn_predictions, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
  
    cnn_report = classification_report(y_test_classes, cnn_predicted_classes, target_names=['No Depression', 'Depression'])
    print(cnn_report)
    
    return cnn_model, cnn_report

if __name__ == "__main__":
    
    knn_model, knn_report = knn_train(NORNALIZED_DATASET_PATH, 2)
    
    rf_model, rf_report = random_forest_train(NORNALIZED_DATASET_PATH)
    
    cnn_model, cnn_report = cnn_train(NORNALIZED_DATASET_PATH)
    
    print("-----------------------------------------------------")
    print("\nComparison of Models:\n")
    print("KNN:\n", knn_report)
    print("\nRandom Forest:\n", rf_report)
    print("\nConvolutional Neural Network:\n", cnn_report)
    
    cnn_model.save("../best_model.keras")