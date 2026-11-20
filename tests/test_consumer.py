"""
Тесты для стримингового воркера антифрода.
HW02: Streaming Consumer Tests.
"""

import time

from consumer.fraud_consumer import (
    TOPIC_ALERTS,
    TOPIC_APPROVED,
    FraudStreamingService,
    MicroBatchCollector,
    decide_output_topic,
)


def test_microbatch_flush_by_size():
    """Проверка сброса батча при заполнении max_batch_size."""
    collector = MicroBatchCollector(max_batch_size=5, max_latency_ms=1000)
    assert not collector.is_ready()

    for i in range(4):
        collector.add({"order_id": f"ord_{i}"})
    assert not collector.is_ready()

    collector.add({"order_id": "ord_4"})
    assert collector.is_ready()

    batch = collector.flush()
    assert len(batch) == 5
    assert not collector.is_ready()


def test_microbatch_flush_by_timeout():
    """Проверка сброса батча по таймауту ожидания."""
    collector = MicroBatchCollector(max_batch_size=100, max_latency_ms=50)
    collector.add({"order_id": "ord_slow_1"})
    assert not collector.is_ready()

    time.sleep(0.06)
    assert collector.is_ready()

    batch = collector.flush()
    assert len(batch) == 1
    assert not collector.is_ready()


def test_score_batch_fallback():
    """Проверка инференса пачки в режиме fallback (без обученной модели)."""
    service = FraudStreamingService(model=None)
    batch = [{"order_id": "ord_1"}, {"order_id": "ord_2"}]
    scores = service.score_batch(batch)
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_routing_threshold():
    """Скоринг >= 0.70 уходит в alerts, ниже — в approved."""
    assert decide_output_topic(0.70) == TOPIC_ALERTS
    assert decide_output_topic(0.99) == TOPIC_ALERTS
    assert decide_output_topic(0.699) == TOPIC_APPROVED
    assert decide_output_topic(0.0) == TOPIC_APPROVED
