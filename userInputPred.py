import kagglehub
import numpy as np
import pandas as pd
import csv
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')

# Download dataset
path = kagglehub.dataset_download("aasheesh200/framingham-heart-study-dataset")
print("Path to dataset files:", path)

df = pd.read_csv(path + "/framingham.csv")

# Preprocessing
new_df = df.dropna()

X = new_df[['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
            'prevalentStroke', 'prevalentHyp', 'diabetes', 'BMI', 'heartRate']]
Y = new_df[['sysBP', 'diaBP', 'glucose']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train_scaled, Y_train)

# Glucose classification
def classify_glucose(glucose):
    if glucose < 70:
        return "Low"
    elif 70 < glucose < 100:
        return "Normal"
    elif 100 <= glucose < 126:
        return "Medium"
    else:
        return "High"

# BMI Calculation
def calculate_bmi_from_csv(file_path):
    try:
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Extract and validate data
                sex = int(row['sex'])
                age = int(row['age'])
                height_cm = float(row['height'])
                weight_kg = float(row['weight'])
                heart_rate = int(row['heartrate'])

                # Calculate BMI
                height_m = height_cm / 100
                bmi = weight_kg / (height_m ** 2)

                return sex, age, round(bmi, 2), heart_rate
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

# User input and prediction
def get_user_input(file_path, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes):
    print("Reading input from Apple Watch CSV...")
    apple_watch_input = calculate_bmi_from_csv(file_path)

    if not apple_watch_input:
        print("Failed to read Apple Watch input. Exiting.")
        return None

    male, age, BMI, heartRate = apple_watch_input

    print("\nCollecting additional user inputs...")
    # currentSmoker = int(input("Current Smoker (1 for yes, 0 for no): "))
    # cigsPerDay = float(input("Cigarettes per day: ") or 0)  # Default to 0 if no input
    # BPMeds = int(input("Taking Blood Pressure Medications (1 for yes, 0 for no): "))
    # prevalentStroke = int(input("Prevalent Stroke (1 for yes, 0 for no): "))
    # prevalentHyp = int(input("Prevalent Hypertension (1 for yes, 0 for no): "))
    # diabetes = int(input("Diabetes (1 for yes, 0 for no): "))

    # Creating input array for prediction
    input_data = np.array([[male, age, currentSmoker, cigsPerDay, BPMeds,
                            prevalentStroke, prevalentHyp, diabetes, BMI, heartRate]])

    input_data_scaled = scaler.transform(input_data)

    # Model prediction
    prediction = model.predict(input_data_scaled)

    sysBP = float(prediction[0][0])  # Convert Systolic Blood Pressure to normal float
    diaBP = float(prediction[0][1])  # Convert Diastolic Blood Pressure to normal float
    glucose = float(prediction[0][2])  # Convert Glucose Level to normal float


    output_list = [
        male,           # Gender
        age,            # Age
        None,           # Placeholder (Not used in `generate_query`)
        currentSmoker,  # Smoker status
        cigsPerDay,     # Cigarettes per day
        BPMeds,         # Blood Pressure Medications
        prevalentStroke,# Stroke History
        prevalentHyp,   # Hypertension
        diabetes,       # Diabetes
        sysBP,          # Systolic Blood Pressure
        diaBP,          # Diastolic Blood Pressure
        BMI,            # BMI
        heartRate,      # Heart Rate
        glucose         # Glucose
    ]
    # print(output_list)
    return output_list
# Example usage
# results = get_user_input(file_path)
# print("\nOutput Data:", results)