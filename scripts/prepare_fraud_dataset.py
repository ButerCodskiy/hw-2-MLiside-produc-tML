"""
Скрипт подготовки и разметки датасета для задачи Fraud Detection (OlistGuard).
Читает сырые таблицы Olist E-Commerce, рассчитывает признаки на момент оформления заказа
(исключая утечки данных) и формирует целевую метку `is_fraud` на основе реалистичных
комбинаций факторов риска e-commerce с фиксированным seed.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_olist_data_path() -> Path:
    """Определяет путь к сырым CSV файлам Olist."""
    candidates = [
        Path(__file__).resolve().parents[2] / "1_model_train" / "datasets" / "olistbr" / "brazilian-ecommerce" / "versions" / "2",
        Path(__file__).resolve().parents[2] / "4_dbt" / "dbt" / "seeds" / "raw",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "olistbr" / "brazilian-ecommerce" / "versions" / "2",
    ]
    env_cache = os.getenv("KAGGLEHUB_CACHE")
    if env_cache:
        candidates.insert(0, Path(env_cache) / "datasets" / "olistbr" / "brazilian-ecommerce" / "versions" / "2")

    for path in candidates:
        if path.exists() and (path / "olist_orders_dataset.csv").exists():
            return path

    raise FileNotFoundError(
        "Не найден каталог с CSV-файлами Olist.\n"
        "Убедитесь, что датасет скачан через `1_model_train/src/load_dataset.py` "
        "или укажите KAGGLEHUB_CACHE в .env файле."
    )


def prepare_fraud_dataset(data_dir: Path, output_dir: Path, random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает таблицы, рассчитывает признаки и создает выборки train/test.
    """
    np.random.seed(random_seed)
    print(f"[*] Чтение сырых таблиц Olist из: {data_dir}")

    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    payments = pd.read_csv(data_dir / "olist_order_payments_dataset.csv")
    products = pd.read_csv(data_dir / "olist_products_dataset.csv")
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")

    # 1. Агрегация платежей по order_id
    print("[*] Агрегация платежной информации...")
    pay_agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_sequential_count=("payment_sequential", "max"),
        has_credit_card=("payment_type", lambda s: int((s == "credit_card").any())),
        has_voucher=("payment_type", lambda s: int((s == "voucher").any())),
        has_boleto=("payment_type", lambda s: int((s == "boleto").any())),
    ).reset_index()

    # 2. Агрегация позиций заказа
    print("[*] Агрегация позиций и товаров...")
    items_prod = items.merge(products, on="product_id", how="left").merge(sellers, on="seller_id", how="left")
    items_agg = items_prod.groupby("order_id").agg(
        total_items_count=("order_item_id", "count"),
        total_products_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        max_product_weight_g=("product_weight_g", "max"),
        seller_state_first=("seller_state", "first"),
    ).reset_index()

    # 3. Объединение с заказами и покупателями
    print("[*] Джойн с заказами и покупателями...")
    df = orders.merge(customers, on="customer_id", how="inner")
    df = df.merge(pay_agg, on="order_id", how="inner")
    df = df.merge(items_agg, on="order_id", how="inner")

    # Заполнение пропусков
    df["max_product_weight_g"] = df["max_product_weight_g"].fillna(df["max_product_weight_g"].median())
    df["seller_state_first"] = df["seller_state_first"].fillna("UNKNOWN")
    df["customer_state"] = df["customer_state"].fillna("UNKNOWN")

    # Преобразование дат
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_hour"] = df["order_purchase_timestamp"].dt.hour
    df["order_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    # Инженерные признаки
    df["freight_ratio"] = df["total_freight_value"] / (df["total_products_price"] + 0.01)
    df["is_interstate_order"] = (df["customer_state"] != df["seller_state_first"]).astype(int)

    # 4. Расчет риска и целевой метки `is_fraud`
    print("[*] Формирование целевой метки `is_fraud` (антифрод скоринг)...")
    
    # Факторы риска
    # 1) Высокое число рассрочек при относительно небольшой сумме
    risk_installments = (df["max_installments"] >= 10) & (df["total_payment_value"] < 300)
    
    # 2) Экстремально высокая сумма платежа (> 99-го перцентиля)
    high_val_thresh = df["total_payment_value"].quantile(0.99)
    risk_extreme_value = (df["total_payment_value"] > high_val_thresh) & (df["max_installments"] >= 8)

    # 3) Комбинация ваучера и кредитной карты при высокой доле доставки и межрегиональной пересылке
    risk_voucher_combo = (df["has_voucher"] == 1) & (df["has_credit_card"] == 1) & (df["is_interstate_order"] == 1)

    # 4) Аномальная стоимость доставки (доставка дороже товара) при заказе ночью (0..5 утра)
    risk_night_freight = (df["freight_ratio"] > 1.2) & (df["order_hour"].between(0, 5))

    # 5) Множественные последовательные платежи (разбиение суммы)
    risk_split_payments = df["payment_sequential_count"] >= 3

    # Базовая вероятность фрода
    fraud_prob = (
        0.005 +  # фоновый шум
        0.45 * risk_installments.astype(float) +
        0.50 * risk_extreme_value.astype(float) +
        0.35 * risk_voucher_combo.astype(float) +
        0.40 * risk_night_freight.astype(float) +
        0.30 * risk_split_payments.astype(float)
    ).clip(0.002, 0.95)

    # Генерация бинарного таргета
    random_noise = np.random.uniform(0.0, 1.0, size=len(df))
    df["is_fraud"] = (random_noise < fraud_prob).astype(int)

    # Выбор признаков для датасета модели
    feature_cols = [
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_purchase_timestamp",
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
        "is_fraud",
    ]
    clean_df = df[feature_cols].copy()

    # Хронологический split: Train (до 2018), Test (2018 год)
    train_df = clean_df[clean_df["order_purchase_timestamp"] < "2018-01-01"].copy().reset_index(drop=True)
    test_df = clean_df[clean_df["order_purchase_timestamp"] >= "2018-01-01"].copy().reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_fraud.parquet"
    test_path = output_dir / "test_fraud.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("\n" + "=" * 60)
    print(" [✓] Датасет Fraud Detection успешно сформирован!")
    print(f" Train: {len(train_df):,} записей, фрода: {train_df['is_fraud'].sum():,} ({train_df['is_fraud'].mean():.2%}) -> {train_path}")
    print(f" Test:  {len(test_df):,} записей, фрода: {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean():.2%}) -> {test_path}")
    print("=" * 60 + "\n")

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка Olist Fraud Detection датасета")
    parser.add_argument("--data-dir", type=str, default=None, help="Путь к сырым CSV Olist")
    parser.add_argument("--output-dir", type=str, default=None, help="Каталог сохранения Parquet файлов")
    args = parser.parse_args()

    input_path = Path(args.data_dir) if args.data_dir else find_olist_data_path()
    output_path = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "data"

    prepare_fraud_dataset(input_path, output_path)
