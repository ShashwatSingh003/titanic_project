import pandas as pd
import numpy as np
import os

def load_data(path):
    df = pd.read_csv(path)  # Loads the dataset from a CSV file and returns a DataFrame
    return df

def clean_data(df):
    """Cleans the raw data: handles duplicates, missing values, and conversions."""
    df = df.drop_duplicates()  # Remove duplicate rows

    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')  # Convert Fare to numeric type. If conversion fails, set to NaN

    
    df = df.drop(columns=['Cabin'])  # Drop the 'Cabin' column (too many missing values)

    age_median = df.groupby(['Pclass', 'Sex'])['Age'].transform('median') # Impute (fill in) missing Age values based on Pclass and Sex median
    df['Age'] = df['Age'].fillna(age_median)
    
    df['Age'] = df['Age'].fillna(df['Age'].median()) # If some Age values are still missing (group had only NaNs), use overall median

    mode_embarked = df['Embarked'].mode()  # Impute missing Embarked values using mode
    if len(mode_embarked) > 1:
        df['Embarked'] = df['Embarked'].fillna('S')  # Default to 'S'
    else:
        df['Embarked'] = df['Embarked'].fillna(mode_embarked[0])

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Impute missing Fare values with median
    return df
