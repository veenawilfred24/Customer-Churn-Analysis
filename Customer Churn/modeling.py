from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split data into training and testing sets
def split_data(features, target, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Train churn prediction models
def train_models(X_train, y_train):
    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)
    
    # Train Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    
    return log_reg, rf

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return metrics
