import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv("data.csv")

print("First few rows of the dataset:")
print(data.head())

conn = sqlite3.connect('alzheimer_data.db')

cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS alzheimer (
        PatientID INTEGER PRIMARY KEY,
        Age INTEGER,
        Gender INTEGER,
        Ethnicity INTEGER,
        EducationLevel INTEGER,
        BMI REAL,
        Smoking INTEGER,
        AlcoholConsumption REAL,
        PhysicalActivity REAL,
        DietQuality REAL,
        MentalActivity REAL,
        FamilyHistory INTEGER,
        Depression INTEGER,
        Anxiety INTEGER,
        Hypertension INTEGER,
        Diabetes INTEGER,
        HeartDisease INTEGER,
        Stroke INTEGER,
        HighCholesterol INTEGER,
        SleepDisorder INTEGER,
        SocialIsolation INTEGER,
        CognitiveDecline INTEGER,
        VisionLoss INTEGER,
        HearingLoss INTEGER,
        BrainInjury INTEGER,
        Medication INTEGER,
        MemoryComplaints INTEGER,
        BehavioralProblems INTEGER,
        ADL REAL,
        Confusion INTEGER,
        Disorientation INTEGER,
        PersonalityChanges INTEGER,
        DifficultyCompletingTasks INTEGER,
        Forgetfulness INTEGER,
        Diagnosis INTEGER,
        DoctorInCharge TEXT
    )
''')

data.to_sql('alzheimer', conn, if_exists='replace', index=False)

data_from_db = pd.read_sql('SELECT * FROM alzheimer', conn)
print("\nFirst few rows of the dataset from SQLite database:")
print(data_from_db.head())

print("\nData types of the original dataset:")
print(data.dtypes)

print("\nSummary of the original dataset:")
print(data.describe())

print("\nData types of the dataset from SQLite database:")
print(data_from_db.dtypes)

print("\nSummary of the dataset from SQLite database:")
print(data_from_db.describe())

#Check target variable
print("\nUnique values in the target variable:")
print(data['Diagnosis'].value_counts())

#Find the most common value for each feature
print("\nMost common value for each feature:")
common_values = data.mode().iloc[0]
for col in data.columns:
    print(f"{col}: {common_values[col]}")

#Preprocess Features
le = LabelEncoder()
categorical_columns = ['Gender', 'Ethnicity', 'DoctorInCharge']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

#Standardizing numerical features if necessary
scaler = StandardScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

#Data Preprocessing
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

model = RandomForestClassifier(random_state=42)
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)

print("Best parameters found: ", CV_rfc.best_params_)

#Train the final model with the best parameters
model = RandomForestClassifier(**CV_rfc.best_params_, random_state=42)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'\nAccuracy: {accuracy}')
print('Classification Report:')
print(report)

importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

print("\nFeature Importance:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))

cursor.execute('''
    CREATE TABLE IF NOT EXISTS feature_importance (
        Feature TEXT PRIMARY KEY,
        Importance REAL
    )
''')

#Insert feature importance data into the feature_importance table
feature_importance_df.to_sql('feature_importance', conn, if_exists='replace', index=False)

#Verify that the data is stored correctly
feature_importance_from_db = pd.read_sql('SELECT * FROM feature_importance', conn)
print("\nFeature importance stored in SQLite database:")
print(feature_importance_from_db)

conn.close()