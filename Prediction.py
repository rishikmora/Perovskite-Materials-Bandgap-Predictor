# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error

# Load the dataset (Updated path for local machine)
file_path = r"C:\Users\Rishik\Desktop\KGP hck\dataset_excavate.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1')

# Keep a copy of the functional group for later use
df['functional_group'] = df['functional group']

# ----------- DATA PREPROCESSING -----------
# Convert 'PBE band gap' to a binary column (Classification Task)
df['insulator'] = (df['PBE band gap'] >= 0.5).astype(int)

# Identify categorical columns
categorical_cols = ['A', "A'", 'B', "B'", 'functional group']

# Apply Label Encoding to categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ----------- CLASSIFICATION TASK -----------
# Features & Target for Classification
X_class = df.drop(columns=['PBE band gap', 'insulator', 'functional_group'])  # Drop target columns
y_class = df['insulator']

# Split classification dataset
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Keep track of test functional group names
test_functional_groups = df.loc[y_test_c.index, 'functional_group']

# Scale features
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# Train Classification Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_c, y_train_c)

# Predictions & Accuracy
y_pred_c = clf.predict(X_test_c)
accuracy = accuracy_score(y_test_c, y_pred_c)
print(f"Classification Accuracy: {accuracy:.4f}")

# Map predictions to labels
class_labels = {0: "Non-Insulator", 1: "Insulator"}
predicted_classes = [class_labels[p] for p in y_pred_c]

# ----------- REGRESSION TASK -----------
# Filter only insulators (Eg ≥ 0.5 eV)
df_insulators = df[df['PBE band gap'] >= 0.5]

# Features & Target for Regression
X_reg = df_insulators.drop(columns=['PBE band gap', 'insulator', 'functional_group'])  # Drop target columns
y_reg = df_insulators['PBE band gap']

# Split regression dataset
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Keep track of test functional group names for regression
test_functional_groups_reg = df_insulators.loc[y_test_r.index, 'functional_group']

# Scale features
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

# Train Regression Model
reg = XGBRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)

# Predictions & RMSE
y_pred_r = reg.predict(X_test_r)
rmse = root_mean_squared_error(y_test_r, y_pred_r)  # Updated to avoid FutureWarning
print(f"Regression RMSE: {rmse:.4f}")

# ----------- PRINT ONLY INSULATORS' PREDICTED BAND GAPS -----------
# Create a DataFrame with Classification Test Results
classification_results = pd.DataFrame({
    'Functional Group': test_functional_groups.values,
    'Predicted Insulator': predicted_classes
})

# Create a DataFrame with Regression Test Results
regression_results = pd.DataFrame({
    'Functional Group': test_functional_groups_reg.values,
    'Predicted Band Gap': y_pred_r
})

# Merge classification & regression results only for predicted insulators
insulator_results = classification_results[classification_results['Predicted Insulator'] == "Insulator"]
insulator_results = insulator_results.merge(regression_results, on="Functional Group", how="inner")

# Print the filtered results
print("\nFinal Test Results (Only Insulators' Predicted Band Gaps):")
print(insulator_results.to_string(index=False))  # Print only insulators
insulator_results.to_csv("insulator_test_results.csv", index=False)  # Save only insulator results

# ----------- FEATURE IMPORTANCE ANALYSIS -----------
# Feature Importance for Regression Model
feat_importances = pd.Series(reg.feature_importances_, index=X_reg.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features for Band Gap Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()