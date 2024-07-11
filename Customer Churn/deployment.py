# deployment.py
import joblib

# Save the best performing model
def save_model(model, filepath):
    joblib.dump(model, filepath)

# Load a model from a file
def load_model(filepath):
    model = joblib.load(filepath)
    return model

# Integrate the model into a production environment
def score_new_customer(model, customer_data):
    prediction = model.predict([customer_data])
    return prediction

# Monitor model performance over time
# (For simplicity, we'll skip implementing this as it's a more complex setup involving production monitoring tools.)
