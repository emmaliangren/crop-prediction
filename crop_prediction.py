import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Check for missing values
crops.isna().sum()
# Get how many crops there is
crops.crop.unique()

# Split into feature & target sets
X = crops.drop(columns="crop")
y = crops["crop"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=35)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Train a logistic regression model for each feature
features_dict = {}
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled[[feature]], y_train)
    y_pred = log_reg.predict(X_test_scaled[[feature]])

    # Calculate F1 score
    f1 = metrics.f1_score(y_test,y_pred,average="weighted")
    features_dict[feature] = f1
    print(f"F1 score for {feature}: {f1}")

# Obtain the feature with the max F1 score
max = 0
key = ""
for feature in ["N","P","K","ph"]:
    if features_dict[feature]>max:
        max = features_dict[feature]
        key = feature

# Store best predictive feature
best_predictive_feature = {key: features_dict[key]}
print(f"Best Predictive Feature: {best_predictive_feature}")