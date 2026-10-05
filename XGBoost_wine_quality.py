import xgboost as xgb
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid, train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

    print("\nDistributia inainte de SMOTE")
    print(y_train.value_counts().sort_index())
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print("\nDistributia dupa SMOTE")
    print(y_train_resampled.value_counts().sort_index())
    print(f"\nInstante antrenare: {X_train_resampled.shape[0]}, Instante test: {X_test_scaled.shape[0]}")

    # NON SMOTE RETURN
    # return X_train_scaled, X_test_scaled, y_train, y_test
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

X_train_resampled, X_test_scaled, y_train_resampled, y_test = preprocess()

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_resampled.values.ravel())
y_test_encoded = le.transform(y_test.values.ravel())

hyperparameters = {
    'max_depth': [3, 5, 7, 10, 12],
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

results = []

print(f"{len(ParameterGrid(hyperparameters))} combinatii de hiperparametri\n")

for params in ParameterGrid(hyperparameters):
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train_resampled, y_train_encoded) 

    y_pred = model.predict(X_test_scaled)
    
    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    results.append({
        'max_depth': params['max_depth'],
        'n_estimators': params['n_estimators'],
        'learning_rate': params['learning_rate'],
        'F1_Score': f1
    })

df_results = pd.DataFrame(results)

best_model = df_results.loc[df_results['F1_Score'].idxmax()]
print(f"Modelul optim:\n{best_model}")

vmin = df_results['F1_Score'].min()
vmax = df_results['F1_Score'].max()

learning_rates = hyperparameters['learning_rate']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analiza hiperparametrilor', fontsize=16)

axes_flat = axes.flatten()

for i, lr in enumerate(learning_rates):
    df_lr = df_results[df_results['learning_rate'] == lr]
    
    pivot_table = df_lr.pivot(index='max_depth', columns='n_estimators', values='F1_Score')
    
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis", 
                ax=axes_flat[i], vmin=vmin, vmax=vmax, cbar=(i == len(learning_rates)-1))
    
    axes_flat[i].set_title(f'Learning Rate = {lr}')
    axes_flat[i].set_ylabel('max_depth')
    axes_flat[i].set_xlabel('n_estimators')

plt.tight_layout()

best_params = {
    'max_depth': int(best_model['max_depth']),
    'n_estimators': int(best_model['n_estimators']),
    'learning_rate': best_model['learning_rate']
}

best_xgb = xgb.XGBClassifier(**best_params, random_state=42)
best_xgb.fit(X_train_resampled, y_train_encoded)

y_pred_best = best_xgb.predict(X_test_scaled)

print("\nClasificarea celui mai bun model")
print(classification_report(y_test_encoded, y_pred_best, target_names=[str(c) for c in le.classes_], zero_division=0))

cm_best = confusion_matrix(y_test_encoded, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Matrice de Confuzie\n(Optimizat: depth={best_params["max_depth"]}, est={best_params["n_estimators"]}, lr={best_params["learning_rate"]})')
plt.xlabel('Clasa Prezisa')
plt.ylabel('Clasa Reala')


feature_names = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

best_xgb.get_booster().feature_names = feature_names

plt.figure(figsize=(10, 6))
xgb.plot_importance(best_xgb, importance_type='weight', max_num_features=11, height=0.5, ax=plt.gca(), color='teal')
plt.title('Feature Importance')
plt.xlabel('Importanta')
plt.ylabel('Trasaturi')
plt.grid(False)
plt.show()
