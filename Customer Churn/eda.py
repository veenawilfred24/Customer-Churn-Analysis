import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate churn rate
def calculate_churn_rate(df):
    churn_rate = df['Churn'].mean() * 100
    return churn_rate

# Visualize churn distribution across various customer segments
def visualize_churn_distribution(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Churn distribution by State
    sns.countplot(x='State', hue='Churn', data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Churn by State")
    
    # Churn distribution by International Plan
    sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Churn by International Plan")
    
    # Churn distribution by Voice Mail Plan
    sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1, 0])
    axes[1, 0].set_title("Churn by Voice Mail Plan")
    
    # Churn distribution by Customer Service Calls
    sns.countplot(x='Customer service calls', hue='Churn', data=df, ax=axes[1, 1])
    axes[1, 1].set_title("Churn by Customer Service Calls")
    
    plt.tight_layout()
    return fig

# Analyze correlations among features and churn
def analyze_correlations(df):
    corr_matrix = df.corr()
    return corr_matrix
