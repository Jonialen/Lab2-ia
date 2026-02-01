import numpy as np
from data_processor import DataProcessor
from models import LogisticRegression, KNN
from evaluator import ModelEvaluator


def main():
    # Configuración de Semilla para Reproducibilidad
    SEED = 42
    np.random.seed(SEED)

    # Preprocesamiento de datos
    print("\nPreprocesamiento de datos")
    print("-" * 20)

    processor = DataProcessor(
        dataset_path='dataset_phishing.csv',
        output_dir='outputs'
    )

    X_train, X_test, y_train, y_test, feature_names = processor.process_all(random_state=SEED)


    #Regresión Logística
    print("\nRegresión Logística (Manual)")
    print("-" * 20)

    log_reg = LogisticRegression(learning_rate=0.1, epochs=1000)
    log_reg.fit(X_train, y_train)

    # Predicciones
    y_pred_train_lr = log_reg.predict(X_train)
    y_pred_test_lr = log_reg.predict(X_test)

    # Accuracy
    train_acc_lr = np.mean(y_pred_train_lr == y_train)
    test_acc_lr = np.mean(y_pred_test_lr == y_test)

    print(f"\nAccuracy - Regresión Logística:")
    print(f"  Training: {train_acc_lr:.4f}")
    print(f"  Testing: {test_acc_lr:.4f}")

    # Gráficas
    log_reg.plot_loss_curve()
    log_reg.plot_decision_boundary(X_train, y_train, feature_names)

    
    # KNN
    print("\nK-Nearest Neighbors (Manual)")
    print("-" * 20)

    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Predicciones
    y_pred_train_knn = knn.predict(X_train)
    y_pred_test_knn = knn.predict(X_test)

    # Accuracy
    train_acc_knn = np.mean(y_pred_train_knn == y_train)
    test_acc_knn = np.mean(y_pred_test_knn == y_test)

    print(f"\nAccuracy - KNN (k={knn.k}):")
    print(f"  Training: {train_acc_knn:.4f}")
    print(f"  Testing: {test_acc_knn:.4f}")

    # Gráfica
    knn.plot_decision_boundary(X_train, y_train, feature_names)

    
    # ---------------------------------------------------------
    # Task 4: Comparación y Análisis
    # ---------------------------------------------------------
    evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)
    
    # Agregar resultados manuales
    evaluator.add_manual_result("Manual LogReg", y_pred_test_lr)
    evaluator.add_manual_result("Manual KNN (k=3)", y_pred_test_knn)
    
    # Benchmark con Sklearn
    evaluator.benchmark_sklearn(k_neighbors=3, random_state=SEED)
    
    # Mostrar tabla comparativa
    evaluator.display_comparison()


if __name__ == "__main__":
    main()
