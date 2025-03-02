import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r"C:\Users\Prachee\Desktop\Projects\Chatbot\Dataset\mental_health_data.csv")

# Select relevant features
selected_features = ["Academic Pressure", "Sleep Duration", "Financial Stress"]
X = data[selected_features]
y = data["Depression"]  # Updated column name

# Encode Sleep Duration
sleep_mapping = {
    "less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,  # Ideal range
    "more than 8 hours": 4
}
X.loc[:, "Sleep Duration"] = X["Sleep Duration"].map(sleep_mapping)

# Handle missing values
X.fillna(X.median(), inplace=True)  # Fill NaNs with median values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("Models/student_depression_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("Models/student_depression_scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
