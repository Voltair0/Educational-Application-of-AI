import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets

df = pd.concat([X, y], axis=1)


# I **************************************************

print("\n1.Descrierea generala a setului de date si a variabilei tinta:\n")

print(f"Numar de instante: {X.shape[0]}")
print(f"Numar de trasaturi: {X.shape[1]}")
print("\nTipul trasaturilor:")
print(X.dtypes)


# II **************************************************

print("\n2.Distributia Claselor\n")

class_counts = y.value_counts().sort_index()
class_percentages = y.value_counts(normalize=True).sort_index() * 100

dist_df = pd.DataFrame({'Nr instante': class_counts, 'P%': class_percentages})
print(dist_df)

plt.figure(figsize=(8, 5))
sns.countplot(data=y, x=y.columns[0], hue=y.columns[0], legend=False)
plt.title('Distributia calitatii')
plt.xlabel('Calitate')
plt.ylabel('Instante')
# plt.show()


# III **************************************************

print("\n3.Identificarea valorilor lipsa si a problemelor de calitate\n")

missing_values = df.isnull().sum()
missing_pct = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Valori Lipsa': missing_values, 'P%': missing_pct})
print("Valori lipsa pe trasatura:")
print(missing_df[missing_df['Valori Lipsa'] > 0] if missing_df['Valori Lipsa'].sum() > 0 else "Nu avem valori lipsa\n")

duplicates_count = df.duplicated().sum()
duplicates_pct = (duplicates_count / len(df)) * 100
print(f"Instante duplicate: {duplicates_count} ({duplicates_pct:.2f}%)\n")

df_cleaned = df.drop_duplicates()
print(f"Dimensiunea setului de date dupa eliminare: {df_cleaned.shape}")


# IV **************************************************

#a

print("\n4.a)Analiza trasaturilor numerice\n")

X_cleaned = df_cleaned.drop(columns=y.columns)
stats = X_cleaned.describe()
print(stats.round(3))
print("\n")

X_cleaned.hist(bins=20, figsize=(15, 12), color='skyblue', edgecolor='black')
plt.suptitle('Histogramele trasaturilor', fontsize=16)
# plt.show()

Q3 = X_cleaned.quantile(0.75)
Q1 = X_cleaned.quantile(0.25)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_count = ((X_cleaned < lower_bound) | (X_cleaned > upper_bound)).sum()
outliers_pct = (outliers_count / len(X_cleaned)) * 100
outliers_df = pd.DataFrame({
    'Q1 (25%)': Q1,
    'Q3 (75%)': Q3,
    'IQR': IQR,
    'Lower Bound': lower_bound,
    'Upper Bound': upper_bound,
    'Nr outliers': outliers_count,
    'P(%) outliers': outliers_pct
})
print(outliers_df.round(3))
print("\n")

#b 

print("\n4.b)Analiza trasaturilor categoriale\n")

cat_cols = X_cleaned.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    for col in cat_cols:
        print(f"\nTrasatura: {col}")
        print(X_cleaned[col].value_counts(normalize=True) * 100)
else:
    print("Setul de date nu contine trasaturi categoriale\n")


# V **************************************************

print("\n5.Relatia dintre trasaturi si target\n")

corr_matrix = df_cleaned.corr(method='pearson')

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Corelatie Pearson')

target_col = y.columns[0]
correlations_with_target = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)

top_2_features = correlations_with_target.head(2).index.tolist()
print(f"Cele mai informative 2 trăsături: {top_2_features}\n")

for feature in top_2_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_cleaned, x=target_col, y=feature, hue=target_col, legend=False)
    plt.title(f'{feature}({target_col})')
    plt.xlabel('quality')
    plt.ylabel(feature)
    
plt.show()