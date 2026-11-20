"""
Скрипт обучения CatBoost классификатора антифрода.
HW01: CatBoost Fraud Detection Baseline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


FEATURE_COLS = [
    "customer_state",
    "seller_state_first",
    "is_interstate_order",
    "total_items_count",
    "total_products_price",
    "total_freight_value",
    "freight_ratio",
    "total_payment_value",
    "max_installments",
    "payment_sequential_count",
    "has_credit_card",
    "has_voucher",
    "has_boleto",
    "max_product_weight_g",
    "order_hour",
    "order_dayofweek",
]

CAT_FEATURES = ["customer_state", "seller_state_first"]
TARGET_COL = "is_fraud"


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает train и test parquet датасеты.
    """
    base_dir = Path(__file__).resolve().parents[2] / "common" / "data"
    train_path = base_dir / "train_fraud.parquet"
    test_path = base_dir / "test_fraud.parquet"

    if not train_path.exists():
        # Если файлы не сгенерированы, запускаем скрипт подготовки
        from homeworks.common.prepare_fraud_dataset import find_olist_data_path, prepare_fraud_dataset
        raw_dir = find_olist_data_path()
        train_df, test_df = prepare_fraud_dataset(raw_dir, base_dir)
        return train_df, test_df

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def train_fraud_model(
    train_df: pd.DataFrame,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    model_output_path: str = "models/catboost_fraud.cbm",
    cv_folds: int = 5,
) -> CatBoostClassifier:
    """
    Обучает модель с кросс-валидацией и сохраняет артефакт.
    """
    X = train_df[FEATURE_COLS].copy()
    y = train_df[TARGET_COL].copy()

    # Приведение категориальных колонок к строковому типу
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)

    # =========================================================================
    # TODO: [Студент] Рассчитайте коэффициент дисбаланса классов scale_pos_weight
    # Формула: количество отрицательных примеров (0) / количество положительных примеров (1)
    # =========================================================================
    scale_pos_weight: float = ... # TODO: реализуйте расчет

    print(f"[*] Вычисленный scale_pos_weight: {scale_pos_weight}")

    # =========================================================================
    # TODO: [Студент] Проведите StratifiedKFold кросс-валидацию (5 фолдов)
    # 1. Разбейте X, y на 5 стратифицированных фолдов
    # 2. Обучите CatBoostClassifier с параметром scale_pos_weight на train-фолде
    # 3. Рассчитайте PR-AUC и ROC-AUC на val-фолде
    # 4. Выведите средние значения метрик по фолдам
    # =========================================================================
    # TODO: реализуйте кросс-валидацию

    # =========================================================================
    # TODO: [Студент] Обучите финальную модель на всем train_df и сохраните артефакт
    # 1. Создайте экземпляр CatBoostClassifier с параметрами iterations, learning_rate, depth,
    #    cat_features, scale_pos_weight, random_seed
    # 2. Обучите модель на X, y
    # 3. Сохраните артефакт через final_model.save_model(str(model_output_path))
    # =========================================================================
    final_model = None # TODO: обучите и сохраните модель

    return final_model


if __name__ == "__main__":
    train_data, _ = load_dataset()
    train_fraud_model(train_data)
