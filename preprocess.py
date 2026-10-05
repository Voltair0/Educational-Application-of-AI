import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess():
    red_wine = pd.read_csv('winequality-red.csv', sep=';')
    white_wine = pd.read_csv('winequality-white.csv', sep=';')
    df_full = pd.concat([red_wine, white_wine], ignore_index=True)
    df_cleaned = df_full.drop_duplicates()
    X_cleaned = df_cleaned.drop(columns=['quality'])
    y_cleaned = df_cleaned[['quality']]

    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned, 
        y_cleaned, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_cleaned
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    print("\nDistributia inainte de SMOTE")
    print(y_train.value_counts().sort_index())
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print("\nDistributia dupa SMOTE")
    print(y_train_resampled.value_counts().sort_index())
    print(f"\nInstante antrenare: {X_train_resampled.shape[0]}, Instante test: {X_test_scaled.shape[0]}")
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

    # NON-SMOTE RETURN
    # return X_train_scaled, X_test_scaled, y_train, y_test