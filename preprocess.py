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

def engineer_features(df): #Creates new features from existing ones to improve ML performance.
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) # Extract title from name (e.g., Mr, Mrs, Miss)

    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss') # Standardize uncommon titles
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # Create FamilySize = SibSp + Parch + 1 (including self)

    df['IsAlone'] = (df['FamilySize'] == 1).astype(int) # Create IsAlone (1 if alone, 0 if with family)

    df['AgeBin'] = pd.qcut(df['Age'], q=5, duplicates='drop', labels=False) # Bin Age into 5 quantile-based bins (0 to 4)

    df['FareBin'] = pd.qcut(df['Fare'], q=5, duplicates='drop', labels=False) # Bin Fare into 5 quantile-based bins
    return df

def encode_and_scale(df):  #Encodes categorical variables and normalizes numerical ones.
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass', 'Title'], drop_first=False) # One-hot encode Sex, Embarked, Pclass, and Title

    df = df.drop(columns=['Name', 'Ticket', 'PassengerId']) # Drop unneeded columns

    df['Fare'] = np.where(df['Fare'] > df['Fare'].quantile(0.99), df['Fare'].quantile(0.99), df['Fare']) # Cap Fare and Age at 99th percentile to reduce outliers
    df['Age'] = np.where(df['Age'] > df['Age'].quantile(0.99), df['Age'].quantile(0.99), df['Age'])

    df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min()) # Normalize Fare and Age to range [0, 1]
    df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

    return df

def save_outputs(df): #Saves the final cleaned dataset and features to disk.
    os.makedirs('output', exist_ok=True) # Create output/ folder if it doesn't exist
    
    df.to_csv('output/cleaned.csv', index=False) # Save cleaned DataFrame to CSV

    features = df.drop(columns=['Survived']).to_numpy() # Save features (everything except Survived) as .npy
    np.save('output/final_features.npy', features)

    print("✅ Outputs saved: cleaned.csv and final_features.npy")

