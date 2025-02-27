# Drop BloodPressure column
data = data.drop("BloodPressure", axis=1)

# Split the data into features (X) and target (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open("diabetes_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("diabetes_scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
