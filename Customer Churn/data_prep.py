import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Handle missing values
def handle_missing_values(df):
    # Impute numerical columns with mean
    num_imputer = SimpleImputer(strategy='mean')
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute categorical columns with most frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    return df

# Encode categorical variables
def encode_categorical(df):
    le = LabelEncoder()

    # Encode State and other categorical columns
    df['State'] = le.fit_transform(df['State'])
    df['International plan'] = le.fit_transform(df['International plan'])
    df['Voice mail plan'] = le.fit_transform(df['Voice mail plan'])
    
    # Convert 'Churn' to boolean (True/False)
    df['Churn'] = df['Churn'].astype(str).str.lower().map({
        'true': True,
        'false': False,
        'yes': True,
        'no': False
    })
    
    return df

# Feature scaling (optional, depends on the modeling approach)
def scale_features(df):
    scaler = StandardScaler()
    num_cols = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
