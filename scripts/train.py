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
    base_dir = Path(__file__).resolve().parents[1] / "data"
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

    scale_pos_weight: float = y[y==0].shape[0] / y[y==1].shape[0]

    print(f"[*] Вычисленный scale_pos_weight: {scale_pos_weight}")

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=77,
    )

    pr: list[float] = []
    roc: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cb = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            cat_features=CAT_FEATURES,
            depth=depth,
            scale_pos_weight=scale_pos_weight,
            random_seed=77,
            verbose=0,
        )

        cb.fit(X_tr, y_tr)
        preds = cb.predict_proba(X_val)[:, 1]
        roca = roc_auc_score(y_val, preds)
        pra = average_precision_score(y_val, preds)
        roc.append(roca)
        pr.append(pra)

    print(f"[*] CV average ROC-AUC: {sum(roc) / len(roc):.4f}")
    print(f"[*] CV average PR-AUC:  {sum(pr) / len(pr):.4f}")

    final_model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        cat_features=CAT_FEATURES,
        depth=depth,
        scale_pos_weight=scale_pos_weight,
        random_seed=77,
        verbose=100,
    )

    final_model.fit(X=X, y=y)

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_output_path))
    return final_model


if __name__ == "__main__":
    train_data, _ = load_dataset()
    train_fraud_model(train_data)
