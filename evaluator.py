from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

class ModelEvaluator:
    """Clase para evaluar y comparar modelos manuales vs sklearn."""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []

    def benchmark_sklearn(self, k_neighbors=3, random_state=42):
        """Entrena y evalúa modelos de scikit-learn."""
        # Logistic Regression
        sk_lr = SklearnLR(random_state=random_state)
        sk_lr.fit(self.X_train, self.y_train)
        y_pred_lr = sk_lr.predict(self.X_test)
        self._add_result("Sklearn LogReg", y_pred_lr)

        # KNN
        sk_knn = SklearnKNN(n_neighbors=k_neighbors)
        sk_knn.fit(self.X_train, self.y_train)
        y_pred_knn = sk_knn.predict(self.X_test)
        self._add_result(f"Sklearn KNN (k={k_neighbors})", y_pred_knn)

    def add_manual_result(self, name, y_pred):
        """Agrega resultados de modelos implementados manualmente."""
        self._add_result(name, y_pred)

    def _add_result(self, name, y_pred):
        metrics = {
            'Modelo': name,
            'Acc': accuracy_score(self.y_test, y_pred),
            'Prec': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0)
        }
        self.results.append(metrics)

    def display_comparison(self):
        """Imprime la tabla comparativa de métricas."""
        print("\n" + "=" * 60)
        print(f"{ 'Tabla Comparativa de Métricas (Test Set)':^60}")
        print("=" * 60)
        print(f"{ 'Modelo':<25} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8}")
        print("-" * 60)
        
        for res in self.results:
            print(f"{res['Modelo']:<25} | {res['Acc']:.4f}   | {res['Prec']:.4f}   | {res['Recall']:.4f}")
        print("-" * 60)
