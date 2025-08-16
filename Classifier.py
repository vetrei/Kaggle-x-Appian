# 📦 Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 📥 Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# 🧹 Feature Engineering for train set
train_df['Dt_Customer'] = pd.to_datetime(train_df['Dt_Customer'], errors='coerce')
train_df['Customer_Tenure_Days'] = (train_df['Dt_Customer'].max() - train_df['Dt_Customer']).dt.days
train_df['Income'].fillna(train_df['Income'].median(), inplace=True)
train_df['TotalAcceptedCmp'] = train_df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
train_df['TotalChildren'] = train_df['Kidhome'] + train_df['Teenhome']
train_df['Age'] = 2025 - train_df['Year_Birth']
train_df['Spending'] = train_df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Encode categorical columns
categorical_cols = ['Education', 'Marital_Status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# 🧽 Fill all remaining NaNs in train
train_df.fillna(train_df.median(numeric_only=True), inplace=True)

# 🎯 Prepare train features and target
drop_cols = ['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue',
             'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
             'AcceptedCmp4', 'AcceptedCmp5', 'Kidhome', 'Teenhome']
X = train_df.drop(columns=drop_cols + ['Target'])
y = train_df['Target']

# 🔀 Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ⚙️ Define Stacking Model
base_learners = [
    ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
]
meta_learner = LogisticRegression()

stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    passthrough=True,
    cv=5
)

# 🧠 Train model
stacked_model.fit(X_train, y_train)

# 📈 Evaluate
y_pred_stack = stacked_model.predict(X_val)
print("✅ Stacked Accuracy:", accuracy_score(y_val, y_pred_stack))
print("✅ Stacked F1 Score:", f1_score(y_val, y_pred_stack))

# 🧹 Preprocess test set
test_df['Dt_Customer'] = pd.to_datetime(test_df['Dt_Customer'], errors='coerce')
test_df['Customer_Tenure_Days'] = (train_df['Dt_Customer'].max() - test_df['Dt_Customer']).dt.days
test_df['Income'].fillna(train_df['Income'].median(), inplace=True)
test_df['TotalAcceptedCmp'] = test_df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
test_df['TotalChildren'] = test_df['Kidhome'] + test_df['Teenhome']
test_df['Age'] = 2025 - test_df['Year_Birth']
test_df['Spending'] = test_df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Handle unseen categories in test
for col in categorical_cols:
    known = list(label_encoders[col].classes_)
    test_df[col] = test_df[col].apply(lambda x: x if x in known else 'Other')
    if 'Other' not in label_encoders[col].classes_:
        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'Other')
    test_df[col] = label_encoders[col].transform(test_df[col])

# 🧽 Fill NaNs in test
test_df.fillna(train_df.median(numeric_only=True), inplace=True)

# 🔮 Predict on test set
X_test = test_df.drop(columns=drop_cols)
test_df['Target'] = stacked_model.predict(X_test)

# 📝 Save submission
submission = test_df[['ID', 'Target']]
submission.to_csv("final_submission_stacked.csv", index=False)
print("📁 Saved: final_submission_stacked.csv")
