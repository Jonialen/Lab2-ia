import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#------------------------------
#1. Cargar y limpieza

df = pd.read_csv('dataset_phishing.csv')

# Eliminar columna 'url' 
df = df.drop(columns=['url'])

# Identificar columnas categoricas
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Codificar cada variable
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

#------------------------------
#2. Seleccion de features

X = df.drop(columns=['status'])
y = df['status']

# Calcular correlación y seleccionar top 2
correlations = X.corrwith(y).abs().sort_values(ascending=False)
top_2_features = correlations.head(2).index.tolist()

print(f"Features seleccionadas:")
print(f"- {top_2_features[0]} (corr: {correlations[top_2_features[0]]:.3f})")
print(f"- {top_2_features[1]} (corr: {correlations[top_2_features[1]]:.3f})")

# Crear el conjunto de datos con las 2 features
X_2d = X[top_2_features].copy()

#------------------------------
# 3. Escalado

scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

#------------------------------
#4. Split 80/20

X_train, X_test, y_train, y_test = train_test_split(
    X_2d_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# Crear directorio outputs si no existe
import os
os.makedirs('outputs', exist_ok=True)

np.save('outputs/X_train.npy', X_train)
np.save('outputs/X_test.npy', X_test)
np.save('outputs/y_train.npy', y_train)
np.save('outputs/y_test.npy', y_test)

with open('outputs/feature_names.txt', 'w') as f:
    f.write(f"{top_2_features[0]}\n{top_2_features[1]}")

# Scatter plot 2D
plt.figure(figsize=(10, 8))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.5)
plt.xlabel(top_2_features[0])
plt.ylabel(top_2_features[1])
plt.title('Datos de Training en 2D')
plt.colorbar(label='0=Legitimate, 1=Phishing')
plt.savefig('outputs/datos_2d.png', dpi=150, bbox_inches='tight')
plt.close()

