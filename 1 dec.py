#%%
# import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, classification_report, confusion_matrix, 
                             roc_curve, auc, f1_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
import shap
import joblib
import xgboost as xgb
import missingno as msno
from sklearn.feature_selection import RFE

# Load dataset
df = pd.read_csv('filtered_data.csv')
print("Initial dataset shape:", df.shape)

# Step 1: Data Exploration and Initial Checks
print("Initial missing values:\n", df.isnull().sum())
df.info()

# Visualize Missing Data
msno.matrix(df)
plt.show()

# Step 2: Drop duplicates
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# Step 3: Handle invalid values (replace negative values with 0)
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].mask(df[numeric_columns] < 0, 0)
print(f"Shape after replacing invalid values: {df.shape}")

# Step 4: Advanced Missing Value Imputation
# Use KNN imputer for numerical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

# Use most frequent imputation for categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print(f"Shape after imputation: {df.shape}")
print("Missing values after imputation:", df.isnull().sum())

# Step 5: Outlier Detection using Isolation Forest
outlier_detector = IsolationForest(contamination=0.05)  # Adjust contamination if needed
outliers = outlier_detector.fit_predict(df[numeric_cols])
df = df[outliers == 1]  # Keep only the inliers (predicted normal rows)
print(f"Shape after outlier removal: {df.shape}")

# Step 6: Feature Engineering
df['total_casualties'] = df['nkill'].fillna(0) + df['nwound'].fillna(0)
df['region_attack_count'] = df.groupby('region')['attacktype1'].transform('count')

# Step 7: Encoding categorical variables
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    freq_map = df[col].value_counts() / len(df)
    df[col] = df[col].map(freq_map)

# Step 8: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))  # Only scale numeric columns
joblib.dump(scaler, "scaler.pkl")

# Step 9: Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
reduced_data = pca.fit_transform(scaled_features)
print("Explained variance ratio:", np.cumsum(pca.explained_variance_ratio_)[-1])

# Step 10: Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Features")
plt.show()

# Step 11: Save Processed Data
output_file_path = 'processed_cleaned_data.csv'
df.to_csv(output_file_path, index=False)
print(f"Processed data saved to {output_file_path}")

# Step 12: Model Training
features = ['region', 'attacktype1', 'weaptype1', 'nkill', 'nwound', 'total_casualties']
X = df[features]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 13: Handle Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_resampled, y_train_resampled)
y_pred_log = log_reg.predict(X_test)
y_pred_proba_log = log_reg.predict_proba(X_test)[:, 1]

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Train XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Save Models
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")

# Step 14: Model Evaluation
def evaluate_model(y_test, y_pred, y_pred_proba):
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=1),
        "Recall": recall_score(y_test, y_pred, zero_division=1),
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba),
        "F1-Score": f1_score(y_test, y_pred),
        "Matthews Correlation Coefficient": matthews_corrcoef(y_test, y_pred)
    }
    return metrics

# Model Evaluations
metrics_log_reg = evaluate_model(y_test, y_pred_log, y_pred_proba_log)
metrics_rf = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf)
metrics_xgb = evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb)

metrics_df = pd.DataFrame([metrics_log_reg, metrics_rf, metrics_xgb], index=['Logistic Regression', 'Random Forest', 'XGBoost'])
print(metrics_df)

# Step 15: Confusion Matrix & ROC Curve
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

plot_confusion_matrix(y_test, y_pred_log, "Logistic Regression")
plot_roc_curve(y_test, y_pred_proba_log, "Logistic Regression")

plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
plot_roc_curve(y_test, y_pred_proba_rf, "Random Forest")

plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost")
plot_roc_curve(y_test, y_pred_proba_xgb, "XGBoost")

# Step 16: SHAP Analysis for XGBoost
explainer = shap.Explainer(xgb_model, X_train_resampled)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Step 17: Feature Importance - Random Forest
feature_importances_rf = rf.feature_importances_
plt.barh(features, feature_importances_rf)
plt.title("Random Forest Feature Importance")
plt.show()

# Step 18: Recursive Feature Elimination (RFE) for Feature Selection
rfe = RFE(log_reg, n_features_to_select=5)
rfe.fit(X_train_resampled, y_train_resampled)
selected_features = X.columns[rfe.support_]
print(f"Selected features from RFE: {selected_features}")

# Step 19: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=42), param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train_resampled, y_train_resampled)

print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best AUC from GridSearchCV: {grid_search.best_score_}")

# Conclusion
print("Model evaluation completed. Final models and results are saved.")

# %%
