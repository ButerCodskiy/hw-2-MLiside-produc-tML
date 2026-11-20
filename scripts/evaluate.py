"""
Скрипт оценки качества модели на тестовой выборке (2018 год).
HW01: Evaluate CatBoost Fraud Detection Model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from train import CAT_FEATURES, FEATURE_COLS, TARGET_COL, load_dataset


def evaluate_model(
    model_path: str = "models/catboost_fraud.cbm",
    test_df: pd.DataFrame | None = None,
    decision_threshold: float = 0.50,
) -> dict:
    """
    Загружает модель, рассчитывает PR-AUC, ROC-AUC и метрики классификации.
    """
    if test_df is None:
        _, test_df = load_dataset()

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден по пути: {path}")

    model = CatBoostClassifier()
    model.load_model(str(path))

    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[TARGET_COL].copy()

    for col in CAT_FEATURES:
        X_test[col] = X_test[col].astype(str)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Расчет основных метрик
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # =========================================================================
    # TODO: [Студент] Подберите порог отсечения (threshold), обеспечивающий
    # Precision >= 0.70 при максимальном возможном Recall.
    # 1. Используйте precision_recall_curve(y_test, y_proba)
    # 2. Найдите индекс с precision >= 0.70 и максимальным recall
    # 3. Сохраните optimal_threshold, optimal_precision, optimal_recall
    # =========================================================================
    optimal_threshold = decision_threshold
    optimal_precision = 0.0
    optimal_recall = 0.0
    # TODO: реализуйте подбор оптимального порога

    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(" [✓] РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ НА ХОЛДАУТЕ 2018 ГОДА")
    print(f" PR-AUC:  {pr_auc:.4f}  (требуется >= 0.30)")
    print(f" ROC-AUC: {roc_auc:.4f}  (требуется >= 0.75)")
    print(f" Оптимальный порог (Precision >= 0.70): {optimal_threshold:.4f}")
    print(f"   - Precision на пороге: {optimal_precision:.4f}")
    print(f"   - Recall на пороге:    {optimal_recall:.4f}")
    print("\nМатрица ошибок (Confusion Matrix):")
    print(f"   TN: {cm[0, 0]:<6} | FP: {cm[0, 1]}")
    print(f"   FN: {cm[1, 0]:<6} | TP: {cm[1, 1]}")
    print("\nОтчет классификации:")
    print(classification_report(y_test, y_pred, digits=4))
    print("=" * 60 + "\n")

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "optimal_threshold": float(optimal_threshold),
        "optimal_precision": float(optimal_precision),
        "optimal_recall": float(optimal_recall),
    }


if __name__ == "__main__":
    evaluate_model()
