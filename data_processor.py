import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


class DataProcessor:
    """Clase para preprocesar datos de phishing y generar datasets de entrenamiento/prueba."""

    def __init__(self, dataset_path='dataset_phishing.csv', output_dir='outputs'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None

    def load_and_clean(self):
        """Carga y limpia el dataset"""
        self.df = pd.read_csv(self.dataset_path)

        # Eliminar columna 'url'
        self.df = self.df.drop(columns=['url'])

        # Identificar columnas categóricas
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # Codificar cada variable
        for col in categorical_cols:
            self.df[col] = pd.factorize(self.df[col])[0]

        print("Dataset cargado")

    def select_features(self, n_features=2):
        """Selecciona las top N features basadas en correlación con la variable objetivo."""
        X = self.df.drop(columns=['status'])
        y = self.df['status']

        # Calcular correlación y seleccionar top features
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        self.feature_names = correlations.head(n_features).index.tolist()

        print(f"\nFeatures seleccionadas:")
        for feat in self.feature_names:
            print(f"- {feat} (corr: {correlations[feat]:.3f})")

        return X[self.feature_names].copy(), y

    def scale_and_split(self, X, y, test_size=0.2, random_state=42):
        """Escala las features y divide en train/test."""
        # Escalado
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split 80/20
        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Convertir a numpy arrays para evitar problemas con índices de pandas
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

        print(f"\nSplit completado:")
        print(f"- Train: {len(self.X_train)} muestras")
        print(f"- Test: {len(self.X_test)} muestras")

    def save_data(self):
        """Guarda los datos procesados en archivos."""
        os.makedirs(self.output_dir, exist_ok=True)

        np.save(f'{self.output_dir}/X_train.npy', self.X_train)
        np.save(f'{self.output_dir}/X_test.npy', self.X_test)
        np.save(f'{self.output_dir}/y_train.npy', self.y_train)
        np.save(f'{self.output_dir}/y_test.npy', self.y_test)

        with open(f'{self.output_dir}/feature_names.txt', 'w') as f:
            f.write('\n'.join(self.feature_names))

        print(f"\nDatos guardados en '{self.output_dir}/'")

    def plot_data(self):
        """Genera scatter plot 2D de los datos de entrenamiento."""
        plt.figure(figsize=(10, 8))
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1],
                   c=self.y_train, cmap='coolwarm', alpha=0.5)
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])
        plt.title('Datos de Training en 2D')
        plt.colorbar(label='0=Legitimate, 1=Phishing')
        plt.savefig(f'{self.output_dir}/datos_2d.png', dpi=150, bbox_inches='tight')
        plt.close()

    def process_all(self, random_state=42):
        """Ejecuta todo el pipeline de procesamiento."""
        self.load_and_clean()
        X, y = self.select_features(n_features=2)
        self.scale_and_split(X, y, random_state=random_state)
        self.save_data()
        self.plot_data()

        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names
