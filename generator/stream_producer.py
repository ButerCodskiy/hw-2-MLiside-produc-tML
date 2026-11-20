"""
Генератор потока транзакций в Kafka.
HW02: Stream Producer.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from aiokafka import AIOKafkaProducer
import pandas as pd


TOPIC_INPUT = os.getenv("TOPIC_INPUT", "olist.orders.stream")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def load_test_records() -> list[dict]:
    """Загружает тестовые транзакции для генерации."""
    data_path = Path(__file__).resolve().parents[2] / "common" / "data" / "test_fraud.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
            df[col] = df[col].astype(str)
    else:
        # Fallback на синтетические тестовые строки
        df = pd.DataFrame([{
            "order_id": f"ord_stream_{i}",
            "customer_id": f"cust_{i}",
            "customer_state": "SP",
            "seller_state_first": "RJ",
            "is_interstate_order": 1,
            "total_items_count": 2,
            "total_products_price": 250.0 + (i * 10 % 500),
            "total_freight_value": 30.0,
            "freight_ratio": 0.12,
            "total_payment_value": 280.0 + (i * 10 % 500),
            "max_installments": (i % 12) + 1,
            "payment_sequential_count": 1,
            "has_credit_card": 1,
            "has_voucher": 0,
            "has_boleto": 0,
            "max_product_weight_g": 800.0,
            "order_hour": 14,
            "order_dayofweek": 3,
        } for i in range(500)])
    return df.to_dict(orient="records")


async def produce_stream(rps: int, duration_sec: int):
    records = load_test_records()
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )
    await producer.start()
    print(f"[*] Генерация потока в топик '{TOPIC_INPUT}' с RPS={rps} в течение {duration_sec}c...")

    interval = 1.0 / rps
    start_time = time.time()
    total_sent = 0
    idx = 0

    try:
        while time.time() - start_time < duration_sec:
            record = records[idx % len(records)].copy()
            # Обеспечиваем уникальность order_id для стрима
            record["order_id"] = f"{record.get('order_id', 'ord')}_{int(time.time()*1000)}"
            await producer.send(TOPIC_INPUT, record)
            total_sent += 1
            idx += 1
            await asyncio.sleep(interval)
    finally:
        await producer.stop()

    elapsed = time.time() - start_time
    print(f"[✓] Отправлено {total_sent} сообщений за {elapsed:.2f} сек (фактический RPS: {total_sent/elapsed:.1f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Stream Producer для OlistGuard")
    parser.add_argument("--rps", type=int, default=50, help="Запросов в секунду")
    parser.add_argument("--duration", type=int, default=15, help="Длительность генерации в секундах")
    args = parser.parse_args()

    asyncio.run(produce_stream(args.rps, args.duration))
