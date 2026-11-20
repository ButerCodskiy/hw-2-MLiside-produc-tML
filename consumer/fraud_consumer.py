"""
Потоковый Kafka Consumer с поддержкой микробатчинга для скоринга антифрода.
HW02: Kafka Consumer with Micro-batching.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from catboost import CatBoostClassifier
import pandas as pd

logger = logging.getLogger("fraud_consumer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TOPIC_INPUT = os.getenv("TOPIC_INPUT", "olist.orders.stream")
TOPIC_ALERTS = os.getenv("TOPIC_ALERTS", "olist.fraud.alerts")
TOPIC_APPROVED = os.getenv("TOPIC_APPROVED", "olist.orders.approved")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
MODEL_PATH = Path("models/catboost_fraud.cbm")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", "50"))
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.70"))


def decide_output_topic(prob: float, threshold: float = FRAUD_THRESHOLD) -> str:
    """
    Маршрутизация: fraud.alerts при score >= threshold, иначе orders.approved.
    """
    # =========================================================================
    # TODO: [Студент] Верните TOPIC_ALERTS если prob >= threshold, иначе TOPIC_APPROVED
    # =========================================================================
    raise NotImplementedError("Реализуйте decide_output_topic")


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


class MicroBatchCollector:
    """
    Буфер сбора сообщений для микробатчинга по размеру или времени ожидания.
    """
    def __init__(self, max_batch_size: int, max_latency_ms: int):
        self.max_batch_size = max_batch_size
        self.max_latency_sec = max_latency_ms / 1000.0
        self.buffer: list[dict[str, Any]] = []
        self.last_flush_time = time.time()

    def add(self, item: dict[str, Any]):
        self.buffer.append(item)

    def is_ready(self) -> bool:
        # =====================================================================
        # TODO: [Студент] Верните True, если:
        # 1. Размер буфера достиг max_batch_size
        # ИЛИ
        # 2. Буфер не пуст и прошло больше max_latency_sec с последнего сброса
        # =====================================================================
        # TODO: реализуйте проверку готовности пачки
        return False

    def flush(self) -> list[dict[str, Any]]:
        batch = self.buffer
        self.buffer = []
        self.last_flush_time = time.time()
        return batch


class FraudStreamingService:
    def __init__(self, model: CatBoostClassifier | None = None):
        self.model = model
        self.collector = MicroBatchCollector(BATCH_SIZE, MAX_LATENCY_MS)

    def load_model(self):
        if MODEL_PATH.exists():
            logger.info("Загрузка CatBoost модели из %s...", MODEL_PATH)
            self.model = CatBoostClassifier()
            self.model.load_model(str(MODEL_PATH))
        else:
            logger.warning("Модель %s не найдена! Используется заглушка для тестов.", MODEL_PATH)

    def score_batch(self, batch: list[dict[str, Any]]) -> list[float]:
        """
        Векторизованный инференс пачки заказов.
        """
        if not batch:
            return []

        if self.model is None:
            # Заглушка, если модель еще не обучена
            return [0.05] * len(batch)

        df = pd.DataFrame(batch)
        features_df = df[FEATURE_COLS].copy()
        for c in ["customer_state", "seller_state_first"]:
            if c in features_df.columns:
                features_df[c] = features_df[c].astype(str)

        probas = self.model.predict_proba(features_df)[:, 1]
        return [float(p) for p in probas]

    async def run(self):
        self.load_model()
        consumer = AIOKafkaConsumer(
            TOPIC_INPUT,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id="fraud-scoring-group",
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda m: json.dumps(m, default=str).encode("utf-8"),
        )

        await consumer.start()
        await producer.start()
        logger.info("[✓] Kafka Consumer и Producer запущены. Ожидание событий...")

        try:
            while True:
                # Читаем доступные сообщения коротким таймаутом
                msg_dict = await consumer.getmany(timeout_ms=10, max_records=BATCH_SIZE)
                for tp, messages in msg_dict.items():
                    for msg in messages:
                        self.collector.add(msg.value)

                # =============================================================
                # TODO: [Студент] Если микробатч готов, выполните скоринг
                # 1. Извлеките накопленную пачку через batch = self.collector.flush()
                # 2. Выполните скоринг: probas = self.score_batch(batch)
                # 3. Для каждого элемента сформируйте событие и отправьте:
                #    - в TOPIC_ALERTS при prob >= FRAUD_THRESHOLD
                #    - в TOPIC_APPROVED при prob < FRAUD_THRESHOLD
                # =============================================================
                if self.collector.is_ready():
                    pass # TODO: реализуйте скоринг и отправку сообщений

        except asyncio.CancelledError:
            logger.info("Остановка consumer по сигналу...")
        finally:
            await consumer.stop()
            await producer.stop()
            logger.info("Kafka Consumer и Producer остановлены.")


if __name__ == "__main__":
    service = FraudStreamingService()
    asyncio.run(service.run())
