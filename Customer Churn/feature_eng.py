import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create additional features
def create_features(df):
    # Create tenure categories from 'Account length'
    bins = [0, 50, 100, 150, 200, 250]
    labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    df['Tenure Category'] = pd.cut(df['Account length'], bins, labels=labels)

    # Convert 'Tenure Category' to a numerical format using LabelEncoder
    le = LabelEncoder()
    df['Tenure Category'] = le.fit_transform(df['Tenure Category'])

    # Total minutes and total charges
    df['Total minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes']
    df['Total charges'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge']

    return df
# Select features for modeling
def select_features(df):
    features = ['State', 'Area code', 'International plan', 'Voice mail plan', 'Customer service calls', 
                'Tenure Category', 'Total minutes', 'Total charges']
    target = 'Churn'
    return df[features], df[target]
