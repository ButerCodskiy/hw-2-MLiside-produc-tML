# Пошаговый план работы (`PLAN.md`) — HW02 OlistGuard

Этот файл — твой рабочий чеклист по этапам: какие команды запускать в терминале, что писать в коде и как проверять каждый чекпоинт. Общее описание задачи и словарик терминов — в [`SPEC.md`](SPEC.md). Правила нашего взаимодействия (с `/grill-me` перед каждой новой технологией и senior-ревью) — в [`GEMINI.md`](GEMINI.md).

---

## ООП в этом проекте: где оно полезно, а где — оверинжиниринг

Освежить ООП на реальном сервисе — хорошая идея. Главное правило продакшена: **класс нужен там, где есть состояние (state) или управляемый ресурс (соединение, модель, метрики)**. Там, где логика без состояния, достаточно обычной функции.

- **Где ООП здесь к месту (и реально используется в проде):**
  1. `ServiceConfig` (`@dataclass` или Pydantic) — хранит все настройки из переменных окружения (`TOPIC_INPUT`, `BATCH_SIZE`, `FRAUD_THRESHOLD` и т.д.) в одном объекте вместо десятка глобальных переменных.
  2. `MicroBatchCollector` — уже есть в стартере; инкапсулирует состояние буфера (`buffer`, таймер последнего сброса).
  3. `FraudScorer` (опционально) — оборачивает загрузку и вызов `CatBoostClassifier`; удобно подменять фейковым объектом в тестах (**Dependency Injection**).
  4. `MetricsTracker` — копит количество обработанных заказов и список задержек, умеет считать RPS и `p95`.
  5. `FraudStreamingService` — управляет жизненным циклом воркера (`start`, основной цикл, `graceful shutdown`, `commit`).
- **Где будет оверинжиниринг (так делать не нужно):**
  - Создавать абстрактные фабрики и иерархии классов ради `decide_output_topic` (это одна строчка `if/else`, плюс её напрямую импортирует стартовый тест `tests/test_consumer.py`).
  - Ломать имена и сигнатуры `decide_output_topic`, `MicroBatchCollector`, `FraudStreamingService` — их проверяют автотесты из стартера.

---

## Этап 0. Гигиена Git, терминал, данные из `my_1_hw` и модель `CatBoost`

> **Статус:** `[ ]` в работе  
> **Гейт:** теория по `uv`, `Path`, `os.getenv`, `.gitignore` разобрана; перед обучением `CatBoost` сделаем короткий чек по `scale_pos_weight` и `StratifiedKFold`.

### Шаги
1. **Починить корневой `.gitignore`:**
   - Убрать опечатку `homework/` → `homeworks_etalon/`, добавить `my_1_hw/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.env`, `*.parquet`, `*.cbm`.
   - Проверить в терминале из корня `/home/barshay/projects/hw-2`:
     ```bash
     git status
     ```
2. **Сгенерировать датасет фрода (`train_fraud.parquet` и `test_fraud.parquet`) из сырых CSV в `my_1_hw`:**
   - В терминале перейти в `homeworks/common`, поставить зависимости и запустить скрипт с указанием пути к CSV:
     ```bash
     cd homeworks/common
     uv sync
     uv run python prepare_fraud_dataset.py --data-dir ../../my_1_hw/mlops/data
     ```
   - Проверить, что файлы появились:
     ```bash
     ls -lh data/
     ```
3. **Обучить `CatBoostClassifier` (закрыть 3 TODO в `homeworks/hw01_model_train_fastapi/src/train.py`):**
   - Перейти в `homeworks/hw01_model_train_fastapi` и поставить зависимости:
     ```bash
     cd ../hw01_model_train_fastapi
     uv sync
     ```
   - Дописать в `src/train.py`: расчёт `scale_pos_weight`, `StratifiedKFold` (5 фолдов, `PR-AUC` и `ROC-AUC`), обучение финальной модели и сохранение в `models/catboost_fraud.cbm` (убедиться, что папка `models/` создаётся через `Path(...).parent.mkdir(parents=True, exist_ok=True)`).
   - Запустить обучение и проверку метрик в терминале:
     ```bash
     uv run python src/train.py
     uv run python src/evaluate.py
     ```
   - Скопировать обученную модель в папку HW02:
     ```bash
     mkdir -p ../hw02_kafka_streaming/models
     cp models/catboost_fraud.cbm ../hw02_kafka_streaming/models/catboost_fraud.cbm
     ```
4. **Подготовить окружение HW02:**
   ```bash
   cd ../hw02_kafka_streaming
   uv sync
   uv run pytest tests/ -v
   ```

### Чекпоинт 0 (проверка в терминале)
- `git status` из корня не показывает `homeworks_etalon/`, `my_1_hw/`, `.venv/`, `*.parquet`, `*.cbm`.
- Существуют `homeworks/common/data/test_fraud.parquet` и `homeworks/hw02_kafka_streaming/models/catboost_fraud.cbm`.
- `uv run pytest tests/ -v` запускается (сейчас 3 теста упадут, 1 пройдёт — это ожидаемо до написания кода).

---

## Этап 1. Docker, сеть `olistguard` и запуск Kafka + Kafka UI

> **Статус:** `[ ]` не начат  
> **Гейт:** перед стартом — `/grill-me` #2 по Docker, портам и сетям.

### Шаги
1. Разобрать `homeworks/hw02_kafka_streaming/docker-compose.yaml` (понять разницу между `localhost:9092` и `kafka:29092`, `healthcheck`, `depends_on`).
2. В терминале проверить, что Docker-демон запущен:
   ```bash
   docker info
   ```
3. Создать общую Docker-сеть (один раз; она же нужна для HW03):
   ```bash
   docker network create olistguard
   docker network ls | grep olistguard
   ```
4. Поднять Kafka и Kafka UI из папки `homeworks/hw02_kafka_streaming`:
   ```bash
   cd homeworks/hw02_kafka_streaming
   docker compose up -d
   docker compose ps
   ```
5. Посмотреть логи контейнера Kafka (убедиться, что стартовал без ошибок):
   ```bash
   docker compose logs -f --tail=50 kafka
   ```
   *(Выход из просмотра логов: `Ctrl+C` — контейнеры при этом продолжат работать.)*

### Чекпоинт 1 (проверка в терминале и браузере)
- `docker compose ps` показывает оба контейнера (`olistguard-kafka` и `olistguard-kafka-ui`) в статусе `Up` / `healthy`.
- В браузере открывается `http://localhost:8080`, внутри виден живой кластер `olistguard-cluster`.

---

## Этап 2. Теория Kafka, `asyncio` и контракты данных

> **Статус:** `[ ]` не начат  
> **Гейт:** перед стартом — `/grill-me` #3 по Kafka и `asyncio`.

### Шаги
1. Пройти `/grill-me` #3 и разобрать на пальцах:
   - топики, партиции, consumer groups, оффсеты и коммиты;
   - отличие `at-most-once` от `at-least-once`;
   - как работает `event loop` в `asyncio`, почему `await` не блокирует цикл, а тяжёлый синхронный код — блокирует.
2. Посмотреть `homeworks/common/schemas.py` и `consumer/fraud_consumer.py` и утвердить точный JSON-формат сообщений для:
   - `olist.fraud.alerts`
   - `olist.orders.approved`
3. Потренироваться смотреть топики и сообщения прямо из терминала (через `docker exec` внутрь контейнера `olistguard-kafka`):
   ```bash
   docker exec -it olistguard-kafka kafka-topics --bootstrap-server localhost:29092 --list
   ```

### Чекпоинт 2
- Ты можешь своими словами объяснить, почему автокоммит оффсетов в паре с микробатчингом приводит к потере заказов при падении воркера.
- Утверждены схемы (поля) выходных JSON-сообщений и структура классов для Этапа 3.

---

## Этап 3. Реализация `consumer/fraud_consumer.py` (ООП + продакшен-надёжность)

> **Статус:** `[ ]` не начат  
> **Гейт:** перед стартом — `/grill-me` #4 по таймерам (`monotonic`), `asyncio.gather`, ручному коммиту и сигналам остановки.

### Шаги (пишешь сам фрагментами → после каждого шага senior-ревью)
1. **Шаг 3.1 — Конфиг, маршрутизация и буфер:**
   - Вынести настройки в класс/dataclass `ServiceConfig` (с сохранением констант уровня модуля для обратной совместимости с тестами).
   - Реализовать `decide_output_topic(prob, threshold)`.
   - Реализовать `MicroBatchCollector.is_ready()` (размер `>= max_batch_size` ИЛИ непустой буфер + истёк таймаут).
2. **Шаг 3.2 — Батчевый скоринг и отправка:**
   - Выделить чистую обработку одного батча в отдельный метод `FraudStreamingService.process_batch(batch, producer)` (удобно и читать, и тестировать, и вызывать при shutdown).
   - Вызывать `score_batch(batch)` **1 раз на всю пачку** (обсудим вынос CPU-инференса в `asyncio.to_thread`).
   - Формировать выходные события и отправлять всю пачку параллельно с ожиданием подтверждения через `await asyncio.gather(...)`.
3. **Шаг 3.3 — Надёжность (Manual Commit + Graceful Shutdown):**
   - Передать `enable_auto_commit=False` в `AIOKafkaConsumer`.
   - Вызывать `await consumer.commit()` **строго после** того, как брокер подтвердил запись всех сообщений текущего батча.
   - Обработать остановку (`Ctrl+C` / `SIGINT` / `SIGTERM`): дообработать остаток `self.collector.buffer` через тот же `process_batch(...)`, закоммитить оффсеты, и только потом закрыть `consumer` и `producer`.

### Чекпоинт 3 (проверка в терминале)
- Из `homeworks/hw02_kafka_streaming`:
  ```bash
  uv run python consumer/fraud_consumer.py
  ```
- В логах видно: модель `models/catboost_fraud.cbm` загружена (не заглушка), consumer подключился к Kafka, при нажатии `Ctrl+C` корректно пишет лог о дообработке/коммите и завершается с кодом `0` (без необработанного traceback).

---

## Этап 4. Автотесты (`tests/test_consumer.py`)

> **Статус:** `[ ]` не начат  
> **Гейт:** перед стартом — `/grill-me` #5 по `pytest`, `pytest-asyncio` и мокам/фейкам.

### Шаги
1. Запустить 4 базовых теста:
   ```bash
   cd homeworks/hw02_kafka_streaming
   uv run pytest tests/ -v
   ```
2. Дописать в `tests/test_consumer.py` новые тесты (используя наши классы и фейковые объекты вместо реальной Kafka):
   - проверка, что модель (`predict_proba`) вызывается ровно **1 раз на батч** из N заказов;
   - проверка `process_batch`: заказы с `prob >= 0.70` ушли в `olist.fraud.alerts`, с `prob < 0.70` — в `olist.orders.approved`;
   - проверка, что `commit()` вызывается после отправки, и что при остановке остаток буфера не теряется.

### Чекпоинт 4 (проверка в терминале)
- `uv run pytest tests/ -v` — все тесты зелёные (`PASSED`), ноль ошибок и предупреждений о невызванных корутинах.

---

## Этап 5. Нагрузочный прогон (`>= 100 RPS`), метрики `p95` и сверка с эталоном

> **Статус:** `[ ]` не начат  
> **Гейт:** перед стартом — `/grill-me` #6 по замерам RPS, `p95 latency` и точному rate-limiting.

### Шаги
1. Доработать сбор метрик (например, класс `MetricsTracker` в `fraud_consumer.py` + проброс timestamp создания сообщения `produced_at` из `stream_producer.py`) и поправить темп отправки в `stream_producer.py` (чтобы реально выдавал `100 RPS`, а не меньше).
2. Запустить **два терминала** одновременно (оба из `homeworks/hw02_kafka_streaming`):
   - **Терминал 1 (Consumer):**
     ```bash
     uv run python consumer/fraud_consumer.py
     ```
   - **Терминал 2 (Producer — 100 RPS на 30 секунд):**
     ```bash
     uv run python generator/stream_producer.py --rps 100 --duration 30
     ```
3. Проверить результат:
   - В Kafka UI (`http://localhost:8080`) открыть топики `olist.orders.stream`, `olist.fraud.alerts`, `olist.orders.approved` и убедиться, что в обоих выходных топиках есть реальные сообщения.
   - Посмотреть consumer lag (отставание группы `fraud-scoring-group` должно упасть до `0`).
   - Повторить прогон и нажать `Ctrl+C` в Терминале 1 прямо во время потока — убедиться, что остаток буфера дообработался, а после перезапуска consumer дочитал ровно с места остановки.
4. Зафиксировать итоговые цифры (**RPS** и **Latency p95**) и сравнить с синхронным REST API из HW01.
5. **Финальное пост-ревью:** только после всех чекпоинтов открыть `homeworks_etalon/hw02_kafka_streaming/` и сравнить наш код с авторским.

### Чекпоинт 5 (финальная приёмка)
- Прогон `>= 100 RPS` (30 сек) пройден без падений.
- В Kafka UI видны сообщения в `olist.fraud.alerts` и `olist.orders.approved`, lag = `0`.
- Есть конкретные цифры `RPS` и `Latency p95`.
- Остановить контейнеры после работы:
  ```bash
  docker compose down
  ```
