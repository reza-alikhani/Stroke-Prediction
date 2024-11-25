# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Step 2: Load the dataset
file_path = '/Users/reza/Desktop/Courses/Fall 2024/E-health, M-health and telemedicine/Stroke prediction/Datasets/Kaggle Stroke Prediction Dataset/healthcare-dataset-stroke-data.csv'  # Replace with the actual path if needed
data = pd.read_csv(file_path)

# Check for missing values (do not print them)
missing_values = data.isnull().sum()

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Impute numerical columns with the mean
num_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Impute categorical columns with the mode
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Retain 'id' in the dataset but exclude it from features for model training
X = data.drop(columns=['stroke', 'id'], axis=1)  # Features (exclude 'stroke' and 'id')
y = data['stroke']  # Target variable

# Encode the 'ever married' column: Yes -> 1, No -> 0
data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})

# Replace 'children' in 'work_type' with NaN
data['work_type'] = data['work_type'].replace('children', np.nan)

# Replace 'Unknown' in 'smoking_status' with NaN
data['smoking_status'] = data['smoking_status'].replace('Unknown', np.nan)

# Perform one-hot encoding for remaining categorical columns
categorical_columns = ['work_type', 'Residence_type', 'smoking_status']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Check if 'work_type' exists and replace 'children' with NaN
if 'work_type' in X.columns:
    X['work_type'] = X['work_type'].replace('children', np.nan)

# Check if 'smoking_status' exists and replace 'Unknown' with NaN
if 'smoking_status' in X.columns:
    X['smoking_status'] = X['smoking_status'].replace('Unknown', np.nan)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Prepare the data (use NaN for 'children' and 'Unknown')
X = data.drop(columns=['stroke'])  # Features
y = data['stroke']  # Target variable

# Handle 'work_type' = 'children' and 'smoking_status' = 'Unknown' as NaN
if 'work_type' in X.columns:
    X['work_type'] = X['work_type'].replace('children', np.nan)

if 'smoking_status' in X.columns:
    X['smoking_status'] = X['smoking_status'].replace('Unknown', np.nan)

# Encode categorical variables using one-hot encoding for categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Impute missing values with the most frequent strategy for categorical features
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Ensure target variable 'y' is numerical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Feature Importance: Random Forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_imputed, y)
rf_importances = random_forest.feature_importances_

# Feature Importance: Information Gain
info_gain_importances = mutual_info_classif(X_imputed, y, random_state=42)

# Combine the results (average importance scores)
combined_importances = (rf_importances + info_gain_importances) / 2

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest Importance': rf_importances,
    'Information Gain Importance': info_gain_importances,
    'Combined Importance': combined_importances
}).sort_values(by='Combined Importance', ascending=False)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Initialize models
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naïve Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

# Train and evaluate each model
confusion_matrices = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Likelihood of stroke (class 1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    # Print results
    print(f"Model: {name}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Sample Predicted Probabilities (Likelihood of Stroke):")
    print(y_pred_proba[:10])  # Display the first 10 probabilities for class 1
    print("=" * 50)

# Stacking Ensemble Model
stacking_model = StackingClassifier(
    estimators=[
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Naïve Bayes", GaussianNB()),
        ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
        ("KNN", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=5,
)

# Train and evaluate the stacking model
stacking_model.fit(X_train, y_train)

# Predict and evaluate the stacking model
y_pred_stacking = stacking_model.predict(X_test)
y_pred_stacking_proba = stacking_model.predict_proba(X_test)[:, 1]  # Likelihood of stroke

# Confusion matrix for the stacking model
stacking_cm = confusion_matrix(y_test, y_pred_stacking)

print("Stacking Ensemble Model")
print("Confusion Matrix:")
print(stacking_cm)
print("Classification Report:")
print(classification_report(y_test, y_pred_stacking))
print("Sample Predicted Probabilities (Likelihood of Stroke):")
print(y_pred_stacking_proba[:10])  # Display the first 10 probabilities for class 1

def gather_user_inputs():
    # Initialize a dictionary to store user responses
    user_data = {}

    # Ask questions for each feature
    user_data['gender'] = input("What is your gender? (Options: Male, Female, Other): ").strip()
    user_data['age'] = float(input("What is your age? (Enter your age in years): ").strip())
    user_data['hypertension'] = int(input("Do you have hypertension? (Options: 1 for Yes, 0 for No): ").strip())
    user_data['heart_disease'] = int(input("Do you have a history of heart disease? (Options: 1 for Yes, 0 for No): ").strip())
    user_data['ever_married'] = int(input("Have you ever been married? (Options: 1 for Yes, 0 for No): ").strip())
    user_data['work_type'] = input("What is your type of work? (Options: Private, Self-employed, Govt_job, Never_worked, Children): ").strip()
    user_data['Residence_type'] = input("What is your residence type? (Options: Urban, Rural): ").strip()
    user_data['avg_glucose_level'] = float(input("What is your average glucose level? (Enter a numerical value in mg/dL): ").strip())
    user_data['bmi'] = float(input("What is your Body Mass Index (BMI)? (Enter a numerical value): ").strip())
    user_data['smoking_status'] = input("What is your smoking status? (Options: formerly smoked, never smoked, smokes, Unknown): ").strip()

    return user_data

def preprocess_user_data(user_data, model_columns):
    # Convert inputs into a DataFrame
    user_df = pd.DataFrame([user_data])
    
    # Handle NaN for 'work_type' = 'children' and 'smoking_status' = 'Unknown'
    user_df['work_type'] = user_df['work_type'].replace('Children', np.nan)
    user_df['smoking_status'] = user_df['smoking_status'].replace('Unknown', np.nan)

    # One-hot encode categorical columns
    categorical_columns = ['gender', 'work_type', 'Residence_type', 'smoking_status']
    user_df = pd.get_dummies(user_df, columns=categorical_columns, drop_first=True)

    # Add missing columns to match the model's input
    for col in model_columns:
        if col not in user_df.columns:
            user_df[col] = 0  # Add missing columns with default value of 0

    # Reorder columns to match the model's training input
    user_df = user_df[model_columns]

    # Impute NaN values with most frequent strategy
    imputer = SimpleImputer(strategy='most_frequent')
    user_df = pd.DataFrame(imputer.fit_transform(user_df), columns=model_columns)

    return user_df

# Prediction function
def predict_stroke_likelihood(trained_model, user_data, model_columns):
    # Preprocess user data
    processed_data = preprocess_user_data(user_data, model_columns)

    # Predict probability of stroke
    likelihood = trained_model.predict_proba(processed_data)[:, 1][0]  # Probability for class 1 (stroke)
    
    return likelihood

# Gather inputs from the user
user_data = gather_user_inputs()

# Get the column names from the trained model (e.g., X_train.columns)
model_columns = X_train.columns  # Use the same columns as the model was trained on

# Predict likelihood using the stacking model (or any trained model)
stroke_likelihood = predict_stroke_likelihood(stacking_model, user_data, model_columns)

print(f"\nThe likelihood of having a stroke is: {stroke_likelihood * 100:.2f}%")



import joblib

# Save the trained model
joblib.dump(stacking_model, 'stroke_prediction_model.pkl')

# Save the feature columns
joblib.dump(list(X_train.columns), 'model_columns.pkl')

