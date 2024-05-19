import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'credit_scores.csv'
df = pd.read_csv(file_path)

# Select relevant columns (only numerical)
selected_columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt',
                    'Credit_History_Age', 'Monthly_Balance', 'Credit_Score']

df = df[selected_columns]

# Drop rows with null values
df = df.dropna()

# Ensure Credit_Score is treated as a categorical variable
df['Credit_Score'] = df['Credit_Score'].astype('category')

# Separate features and target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Identify numerical columns (all columns in X are now numerical)
numerical_cols = X.columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Bundle preprocessing for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
    ])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions 
y_pred = pipeline.predict(X_test) 

# Evaluate the model 
print("Accuracy:", accuracy_score(y_test, y_pred))

import pickle

# Save the model to disk
with open('credit_score_model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
