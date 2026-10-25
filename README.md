# b1-rag

Backend-сервис вопросов по документу. Сервис принимает PDF или DOCX (до 100 страниц), индексирует его в фоне в Qdrant и отвечает на вопросы **только по тексту документа**. Каждое утверждение ответа подкреплено цитатой с указанием страницы и раздела.

Стек: FastAPI, Celery и Redis, Qdrant (гибридный поиск dense + BM25), LangChain, OpenRouter (LLM, эмбеддинги, реранкер), dishka (DI).

## Запуск

```bash
make env                      # .env из .env.example — указать OPENROUTER_API_KEY
make up                       # api :8000, worker, qdrant :6333, redis :6379
open http://localhost:8000/docs
```

`make help` — список всех команд (логи, тесты, линтеры, локальный запуск).

Прогон примеров из раздела «Примеры» на демо-договоре (`examples/contract.pdf`, 6 страниц):

```bash
make examples                 # ответы сохраняются в examples/results/
```

Локально, без Docker для приложения:

```bash
make install
make infra                    # только qdrant и redis
make run-api                  # в одном терминале
make run-worker               # в другом
```

Тесты:

```bash
make test-unit           # домен, парсеры, use cases и API на фейках портов
make test-integration    # настоящие Qdrant и Redis через testcontainers (нужен Docker)
make lint                # ruff + mypy
```

## API

Базовый префикс — `/api`. Ошибки возвращаются в формате `{"detail": "..."}`. Значения в примерах этого раздела показывают формат; реальные ответы на демо-договоре — в разделе «Примеры».

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/documents` | Загрузка документа и запуск индексации |
| `GET` | `/tasks/{task_id}` | Состояние фоновой задачи индексации |
| `GET` | `/documents` | Список проиндексированных документов |
| `POST` | `/documents/{doc_id}/ask?debug=false` | Ответ на вопрос по документу |
| `GET` | `/health` | Доступность Qdrant и Redis |

### `POST /api/documents` → `202 Accepted`

`multipart/form-data`: `file` — PDF или DOCX (до `MAX_UPLOAD_BYTES`, по умолчанию 50 МБ), `title` — необязательное название (по умолчанию имя файла).

```json
{
  "doc_id": "1e1595a1-24c9-453f-a88e-ded2501b9c29",
  "task_id": "1e1595a1-24c9-453f-a88e-ded2501b9c29",
  "status": "queued",
  "status_url": "http://localhost:8000/api/tasks/1e1595a1-24c9-453f-a88e-ded2501b9c29"
}
```

Возможные ошибки: `413` — файл слишком большой (если `Content-Length` запроса больше лимита, ответ приходит сразу, тело не читается; точная проверка размера — при сохранении файла), `415` — формат не поддерживается, `422` — пустой файл, `503` — брокер недоступен. Документ больше `MAX_PAGES` страниц (по умолчанию 100) принимается, но индексация завершается `failed` с причиной: число страниц известно только после разбора.

### `GET /api/tasks/{task_id}` → `200` | `404`

```json
{
  "task_id": "1e1595a1-…",
  "doc_id": "1e1595a1-…",
  "doc_title": "Договор поставки № 47/2025",
  "filename": "contract.pdf",
  "status": "running",
  "stage": "embedding",
  "progress": 0.52,
  "chunks_total": null,
  "error": null,
  "created_at": "2026-10-25T12:29:41.792129Z",
  "updated_at": "2026-10-25T12:29:43.024232Z"
}
```

- `status`: `queued` → `running` → `done` | `failed`. При временной недоступности внешнего сервиса задача возвращается в `queued` с текстом «Временная ошибка, задача будет повторена: …» и повторяется Celery (`INDEX_TASK_MAX_RETRIES`, паузы 30 → 60 → 120 → 240 с); `failed` — только после последней попытки.
- `stage`: `queued`, `parsing`, `chunking`, `embedding`, `upserting`, `done`.
- `progress` — доля выполнения от 0 до 1.
- При `failed` в поле `error` лежит причина: например, «Не удалось извлечь текст из документа (возможно, это скан без текстового слоя)».

### `GET /api/health` → `200` | `503`

```json
{"status": "ok", "components": {"qdrant": true, "redis": true}}
```

Каждый компонент проверяется с таймаутом 2 с. Если хотя бы один недоступен — `503`, `"status": "unavailable"`, а в лог API пишется, какой компонент отказал. OpenRouter не проверяется: это внешний платный сервис.

### `GET /api/documents` → `200`

В списке только полностью проиндексированные документы. Сведения берутся из payload фрагментов в Qdrant. Документ, который ещё индексируется или упал с ошибкой, виден только через `/tasks/{task_id}`.

```json
{
  "documents": [
    {
      "doc_id": "1e1595a1-…",
      "doc_title": "Договор поставки № 47/2025",
      "filename": "contract.pdf",
      "chunks_total": 5,
      "pages_total": 6,
      "indexed_at": "2026-10-25T12:30:02.118Z"
    }
  ],
  "total": 1
}
```

### `POST /api/documents/{doc_id}/ask` → `200` | `404` | `422` | `502` | `503`

`doc_id` и `task_id` в путях — UUID, иначе `422`. `502` — модель дважды подряд вернула ответ не по схеме, `503` — внешний сервис недоступен после повторов.

Тело запроса: `{"question": "…"}` (от 1 до 2000 символов). Необязательный query-параметр `debug=true` добавляет в ответ диагностику поиска.

```json
{
  "answer": "Штраф за просрочку поставки составляет 0,1% от стоимости непоставленного в срок Товара за каждый день просрочки, но не более 10% от общей стоимости Товара.",
  "found": true,
  "citations": [
    {
      "text": "4.2.1. За нарушение сроков поставки Поставщик уплачивает Покупателю\nнеустойку в размере 0,1% от стоимости непоставленного в срок Товара за\nкаждый день просрочки, но не более 10% от общей стоимости Товара.",
      "page": 3,
      "page_to": 3,
      "section": "4.2 Штрафные санкции",
      "chunk_index": 2,
      "char_start": 1853,
      "char_end": 2057
    }
  ],
  "diagnostics": null
}
```

- `citations[].text` — дословная подстрока документа. Модель её не генерирует: см. раздел «Защита от ответов не по документу».
- `page`, `page_to` и `section` вычисляются по позиции самой цитаты, а не по началу фрагмента. Поле `page` равно `null`, если формат не даёт страниц (DOCX без маркеров).
- `char_start` и `char_end` — смещения цитаты в полном тексте документа.

Если ответа в документе нет: `found=false`, `citations=[]`, `answer` — «В документе нет сведений, необходимых для ответа на этот вопрос.»

`diagnostics` при `debug=true`:

```json
{
  "dense_hits": 5,
  "sparse_hits": 5,
  "fused_candidates": 5,
  "reranked_candidates": 5,
  "context_fragments": 5,
  "used_in_answer": [
    2
  ],
  "llm_called": true,
  "reranked": [
    {
      "chunk_index": 2,
      "page_from": 2,
      "page_to": 3,
      "section": "3. СРОКИ И ПОРЯДОК ПОСТАВКИ",
      "rerank_score": 0.3832,
      "dense_rank": 1,
      "sparse_rank": 1,
      "used_in_context": true
    }
  ]
}
```

В `reranked` показан первый из пяти фрагментов, полный ответ — в примере 5 раздела «Примеры».

| Поле | Значение |
|---|---|
| `dense_hits` / `sparse_hits` | сколько фрагментов дала каждая ветвь поиска |
| `reranked[].dense_rank` / `sparse_rank` | место фрагмента в каждой ветви до слияния; `null` — ветвь его не нашла. Показывает, какая ветвь вытащила фрагмент: для точных обозначений обычно BM25, для переформулировок — плотная |
| `fused_candidates` | кандидатов после слияния ветвей (вход реранкера) |
| `reranked_candidates` | кандидатов после переранжирования |
| `context_fragments` | фрагментов, прошедших порог `RERANK_MIN_SCORE` и переданных LLM |
| `used_in_answer` | `chunk_index` фрагментов, на которые опирается ответ |
| `llm_called` | вызывалась ли LLM (без релевантного контекста — нет) |

## Архитектура

Луковичная архитектура. Зависимости направлены внутрь, внешние сервисы стоят за портами (`typing.Protocol`), сборка — в глобальном DI-контейнере dishka (`rag/di.py`). Классы получают в конструкторе только нужные зависимости (роутеры — свои use cases, use cases — порты), а из контейнера достают лишь точки входа: `manage.py start-api` берёт `AppBuilder`, Celery-воркер — `IndexDocument`.

```
rag/
  domain/            сущности, доменные правила, исключения — без зависимостей
    entities.py      Chunk, Document, Answer, Citation, IndexingTask, ParsedDocument…
    layout.py        привязка фрагментов к страницам и разделам по смещениям символов
    grounding.py     проверка ответа LLM на опору в документе
  application/
    ports/           Embeddings, SparseEncoder, VectorStore, Reranker, AnswerGenerator,
                     DocumentParser, Chunker, FileStorage, TaskStatusRepository, TaskQueue
    use_cases/       UploadDocument, IndexDocument, GetIndexingStatus, ListDocuments, AskQuestion
  adapters/
    inbound/api/     FastAPI: роутеры, схемы (Pydantic), обработка ошибок
    inbound/worker/  Celery-воркер — тонкая обёртка над IndexDocument
    outbound/        qdrant, openrouter (llm, embeddings, rerank), fastembed bm25,
                     парсеры pdf/docx, нарезка, файловое хранилище, redis-статусы, celery-очередь
  di.py              провайдеры dishka и глобальный container (общий для API и воркера)
  config.py          все настройки — переменные окружения (pydantic-settings)
manage.py            точки входа: start-api, start-worker
```

**Индексация.** `POST /documents` проверяет формат и размер, сохраняет файл на общий том и записывает статус `queued` в Redis, затем ставит задачу в Celery. Воркер выполняет цепочку: разбор (PyMuPDF / python-docx) → нарезка (LangChain `RecursiveCharacterTextSplitter`) → плотные эмбеддинги (LangChain `OpenAIEmbeddings` через OpenRouter) и BM25 (fastembed) → upsert в Qdrant. Прогресс по этапам пишется в статус.

**Ответ.** Вопрос кодируется в dense- и BM25-вектор. Дальше один запрос `query_points` к Qdrant: две ветви `prefetch` с фильтром по `doc_id` и слияние `FusionQuery` (RRF или DBSF). Затем кандидаты переранжирует реранкер (OpenRouter `/rerank`), фрагменты ниже порога отбрасываются. LLM (LangChain `ChatOpenAI` со структурированным выводом в Pydantic) получает пронумерованные фрагменты, а её ответ проходит проверку в `domain/grounding.py`.

### Qdrant

- Одна коллекция `QDRANT_COLLECTION` на все документы. В каждой точке два именованных вектора: `dense` (cosine, размерность `EMBEDDING_DIM`) и разреженный `bm25` с `modifier=IDF`. IDF считает Qdrant, fastembed даёт только TF-часть, поэтому статистика корректна для всей коллекции.
- Payload фрагмента: `doc_id`, `doc_title`, `filename`, `chunk_index`, `chunks_total`, `pages_total`, `indexed_at`, `page_from`, `page_to`, `section`, `text`, `char_start`, `char_end`, `ready`, а также `page_marks` и `heading_marks` (начала страниц и заголовков внутри фрагмента).
- Payload-индексы: `doc_id` (keyword, `is_tenant=true` — данные документа хранятся рядом), `chunk_index` (по нему строится список документов), `ready` (видимость документа), `section`, `page_from`.
- ID точки — `uuid5(doc_id, chunk_index)`, перед записью старые точки документа удаляются. Повторная доставка задачи не создаёт дублей.
- При старте API и воркера коллекция создаётся, если её нет, и сверяется размерность dense-вектора с `EMBEDDING_DIM`.

### Принятые решения

- **Защита от ответов не по документу.** LLM возвращает `found`, `answer` и ссылки вида «номер фрагмента + дословная цитата». Ссылки на фрагменты вне контекста отбрасываются. Цитата ищется в тексте фрагмента без учёта регистра и пробелов, и в ответ попадает подстрока документа, а не текст модели. Если цитата не нашлась целиком (модель поменяла падеж или пропустила слово), ищется самый длинный непрерывный кусок её слов, не короче двух слов и восьми символов. Цитата, которую не удалось найти и так, считается выдуманной и отбрасывается. `found=true` без единой подтверждённой ссылки превращается в `found=false`. Если после порога реранкера контекст пуст, LLM не вызывается. Промпт запрещает общие знания и объявляет текст фрагментов данными, а не инструкциями (защита от prompt injection в документе). **Ограничение:** проверяется, что каждая цитата настоящая и что ответ на что-то опирается, но не то, что каждое утверждение в тексте `answer` покрыто цитатой — это держится на промпте.
- **Порог реранкера.** На 30 проверочных вопросах (`examples/test_questions.json`) оценка реранкера не отделяет «ответ есть» от «ответа нет»: вопрос без ответа получал до 0,80, а перефразированный вопрос с ответом — 0,098. Поэтому `RERANK_MIN_SCORE=0.05` только отсекает явно нерелевантное и экономит вызов LLM, а решение «нет сведений» принимает LLM с последующей проверкой цитат. Прогон: `uv run python examples/run_test_questions.py`, отчёт — `examples/results/test_questions_report.json`.
- **Точные обозначения в BM25.** Токенизатор BM25 разбивает «4.2» на «4» и «2», а «1 500 000» — на три числа. Поэтому перед кодированием к тексту и к запросу дописываются склеенные токены: `4_2`, `0_1`, `12_03_2024`, `1500000`. Вопрос про «пункт 4.2» находит именно 4.2, а не 2.4. Это проверяется интеграционным тестом на настоящем Qdrant.
- **Страницы и разделы.** PDF разбирается постранично, и за каждой страницей закрепляется диапазон символов. Разделы определяются эвристикой: нумерованные заголовки («4.2 Штрафные санкции»), «Статья/Раздел/Глава/Приложение N», строки капсом. Первая строка перенесённого пункта («1.1. Поставщик … обязуется») заголовком не считается. В DOCX используются стили `Heading`/`Заголовок` и та же эвристика.
- **DOCX и страницы.** В формате DOCX нет страниц. Номера восстанавливаются по маркерам `w:lastRenderedPageBreak`, которые Word пишет при сохранении, а без них — по явным разрывам страниц. Если маркеров нет совсем, возвращается `page = null`.
- **Статусы задач — свои, в Redis.** Модель `IndexingTask` хранится под ключом `rag:task:{id}` с TTL, result backend Celery не используется. Статус записывается до постановки в очередь: у Celery неизвестная задача и задача в очереди одинаково выглядят как `PENDING`, а у нас 404 и `queued` различимы. Плюс есть стадии и прогресс.
- **Async везде.** Use cases и адаптеры асинхронные. В Celery на каждый процесс воркера создаётся один event loop и один DI-контейнер: async-клиенты Qdrant, Redis и httpx привязаны к циклу.
- **Повторы.** Qdrant, реранкер и Redis повторяются через tenacity: экспоненциальная задержка с джиттером, только временные ошибки (сеть, таймаут, 429, 5xx). По умолчанию 6 попыток с паузами 0,5 → 1 → 2 → 4 → 8 с, около 20 секунд — этого хватает, чтобы пережить перезапуск Qdrant; при старте API и воркер так же ждут Qdrant, если он поднимается позже. LLM и эмбеддинги повторяет SDK OpenAI (`LLM_MAX_RETRIES`). Когда попытки исчерпаны, API отвечает `503` с понятным текстом. Ошибки клиента (400, 401) не повторяются. Если сервис недоступен дольше, задача индексации повторяется уже на уровне Celery с паузами от 30 секунд до 10 минут. Celery настроен с `acks_late` и `reject_on_worker_lost`, поэтому упавший воркер не теряет задачу.
- **Видимость документа.** Фрагменты записываются с `ready=false` и невидимы для поиска, `/ask` и списка документов. Когда записаны все фрагменты, одним `set_payload` по фильтру `doc_id` им ставится `ready=true`. Поэтому `/ask` не отвечает по неполному набору фрагментов, а повторная индексация не показывает документ до завершения.
- **Сбой индексации** — статус `failed` с причиной, частично записанные точки удаляются. Исходный файл удаляется после успешной индексации и после окончательного `failed`.
- **Без истории.** Каждый вопрос обрабатывается независимо: ни сессий, ни памяти.

### Настройки

Все параметры — переменные окружения, полный список с комментариями в [`.env.example`](.env.example). Основные:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `OPENROUTER_API_KEY` | — | ключ OpenRouter (LLM, эмбеддинги, реранкер) |
| `LLM_MODEL` | `openai/gpt-4.1-mini` | модель ответа |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `openai/text-embedding-3-large` / `3072` | плотные эмбеддинги |
| `RERANK_MODEL` | `cohere/rerank-v3.5` | реранкер |
| `RERANK_CANDIDATES` | `30` | кандидатов на входе реранкера |
| `RERANK_TOP_N` | `5` | фрагментов на выходе реранкера |
| `RERANK_MIN_SCORE` | `0.05` | порог релевантности для контекста LLM |
| `SEARCH_FUSION` | `rrf` | способ слияния ветвей: `rrf` или `dbsf` |
| `SEARCH_DENSE_LIMIT` / `SEARCH_SPARSE_LIMIT` | `30` / `30` | кандидатов каждой ветви |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | нарезка, символы |

## Примеры

Реальные запросы к демо-договору `examples/contract.pdf` (договор поставки, 6 страниц, 5 фрагментов) и ответы сервиса без правок. Прогон 25.10.2026 на `docker compose`: LLM `openai/gpt-4.1-mini`, эмбеддинги `openai/text-embedding-3-large`, реранкер `cohere/rerank-v3.5` (все через OpenRouter), `SEARCH_FUSION=rrf`, `CHUNK_SIZE=1000`, `RERANK_MIN_SCORE=0.05`.

Повторить: `make up && make examples` — скрипт `examples/run_examples.sh` выполняет те же запросы и сохраняет ответы в `examples/results/`.

### 1. Загрузка документа

```bash
curl -X POST localhost:8000/api/documents -F "file=@examples/contract.pdf" -F "title=Договор поставки № 47/2025"
```

```json
{
  "doc_id": "329ffaf9-8847-4596-8e2e-a2ca40ada226",
  "task_id": "329ffaf9-8847-4596-8e2e-a2ca40ada226",
  "status": "queued",
  "status_url": "http://localhost:8000/api/tasks/329ffaf9-8847-4596-8e2e-a2ca40ada226"
}
```

### 2. Состояние задачи индексации

```bash
curl localhost:8000/api/tasks/329ffaf9-8847-4596-8e2e-a2ca40ada226
```

Индексация заняла около 2,5 секунды.

```json
{
  "task_id": "329ffaf9-8847-4596-8e2e-a2ca40ada226",
  "doc_id": "329ffaf9-8847-4596-8e2e-a2ca40ada226",
  "doc_title": "Договор поставки № 47/2025",
  "filename": "contract.pdf",
  "status": "done",
  "stage": "done",
  "progress": 1.0,
  "chunks_total": 5,
  "error": null,
  "created_at": "2026-10-25T15:38:27.762375Z",
  "updated_at": "2026-10-25T15:38:30.302597Z"
}
```

### 3. Список документов

```bash
curl localhost:8000/api/documents
```

В списке три загрузки одного и того же договора от разных прогонов: каждая загрузка — отдельный документ со своим `doc_id`.

```json
{
  "documents": [
    {
      "doc_id": "329ffaf9-8847-4596-8e2e-a2ca40ada226",
      "doc_title": "Договор поставки № 47/2025",
      "filename": "contract.pdf",
      "chunks_total": 5,
      "pages_total": 6,
      "indexed_at": "2026-10-25T15:38:30.272993Z"
    },
    {
      "doc_id": "6c8b7861-4804-448b-a6f8-8cb4fd70d062",
      "doc_title": "Договор поставки № 47/2025",
      "filename": "contract.pdf",
      "chunks_total": 5,
      "pages_total": 6,
      "indexed_at": "2026-10-25T15:19:09.915584Z"
    },
    {
      "doc_id": "ef605478-49fd-4037-b7dd-6ec86f4b6f48",
      "doc_title": "Договор поставки № 47/2025",
      "filename": "contract.pdf",
      "chunks_total": 5,
      "pages_total": 6,
      "indexed_at": "2026-10-25T15:18:30.111723Z"
    }
  ],
  "total": 3
}
```

### 4. Вопрос по смыслу: «Какой штраф за просрочку поставки?»

Пример из ТЗ. Цитата — дословный текст п. 4.2.1, страница и раздел определены по месту цитаты.

```bash
curl -X POST "localhost:8000/api/documents/329ffaf9-8847-4596-8e2e-a2ca40ada226/ask" \
  -H 'Content-Type: application/json' -d '{"question": "Какой штраф за просрочку поставки?"}'
```

```json
{
  "answer": "Штраф за просрочку поставки составляет 0,1% от стоимости непоставленного в срок Товара за каждый день просрочки, но не более 10% от общей стоимости Товара.",
  "found": true,
  "citations": [
    {
      "text": "4.2.1. За нарушение сроков поставки Поставщик уплачивает Покупателю\nнеустойку в размере 0,1% от стоимости непоставленного в срок Товара за\nкаждый день просрочки, но не более 10% от общей стоимости Товара.",
      "page": 3,
      "page_to": 3,
      "section": "4.2 Штрафные санкции",
      "chunk_index": 2,
      "char_start": 1853,
      "char_end": 2057
    }
  ],
  "diagnostics": null
}
```

### 5. Точное обозначение — номер пункта, с диагностикой: «Что сказано в пункте 4.2.2?»

Нужный фрагмент (`chunk_index=2`) первый в обеих ветвях поиска (`dense_rank=1`, `sparse_rank=1`) и после реранкера. В ответе — именно п. 4.2.2, а не соседние 4.2.1 или 2.4. Раздел фрагмента в диагностике — «3. СРОКИ И ПОРЯДОК ПОСТАВКИ» (с него фрагмент начинается), а у цитаты — «4.2 Штрафные санкции» (где стоит сама цитата).

```bash
curl -X POST "localhost:8000/api/documents/329ffaf9-8847-4596-8e2e-a2ca40ada226/ask?debug=true" \
  -H 'Content-Type: application/json' -d '{"question": "Что сказано в пункте 4.2.2?"}'
```

```json
{
  "answer": "В пункте 4.2.2 сказано, что за нарушение сроков оплаты Покупатель уплачивает Поставщику пени в размере 0,05% от неоплаченной суммы за каждый день просрочки.",
  "found": true,
  "citations": [
    {
      "text": "4.2.2. За нарушение сроков оплаты Покупатель уплачивает Поставщику\nпени в размере 0,05% от неоплаченной суммы за каждый день просрочки.",
      "page": 3,
      "page_to": 3,
      "section": "4.2 Штрафные санкции",
      "chunk_index": 2,
      "char_start": 2059,
      "char_end": 2194
    }
  ],
  "diagnostics": {
    "dense_hits": 5,
    "sparse_hits": 5,
    "fused_candidates": 5,
    "reranked_candidates": 5,
    "context_fragments": 5,
    "used_in_answer": [
      2
    ],
    "llm_called": true,
    "reranked": [
      {
        "chunk_index": 2,
        "page_from": 2,
        "page_to": 3,
        "section": "3. СРОКИ И ПОРЯДОК ПОСТАВКИ",
        "rerank_score": 0.3832,
        "dense_rank": 1,
        "sparse_rank": 1,
        "used_in_context": true
      },
      {
        "chunk_index": 3,
        "page_from": 3,
        "page_to": 5,
        "section": "5. ПРИЁМКА ТОВАРА",
        "rerank_score": 0.1825,
        "dense_rank": 2,
        "sparse_rank": 4,
        "used_in_context": true
      },
      {
        "chunk_index": 1,
        "page_from": 1,
        "page_to": 2,
        "section": "2. ЦЕНА И ПОРЯДОК РАСЧЁТОВ",
        "rerank_score": 0.1135,
        "dense_rank": 3,
        "sparse_rank": 2,
        "used_in_context": true
      },
      {
        "chunk_index": 0,
        "page_from": 1,
        "page_to": 1,
        "section": null,
        "rerank_score": 0.0904,
        "dense_rank": 4,
        "sparse_rank": 3,
        "used_in_context": true
      },
      {
        "chunk_index": 4,
        "page_from": 5,
        "page_to": 5,
        "section": "8. РАЗРЕШЕНИЕ СПОРОВ",
        "rerank_score": 0.0734,
        "dense_rank": 5,
        "sparse_rank": 5,
        "used_in_context": true
      }
    ]
  }
}
```

### 6. Точное обозначение — сумма: «Какова общая стоимость товара и сколько в ней НДС?»

Суммы переданы дословно, в вопросе их нет.

```bash
curl -X POST "localhost:8000/api/documents/329ffaf9-8847-4596-8e2e-a2ca40ada226/ask" \
  -H 'Content-Type: application/json' -d '{"question": "Какова общая стоимость товара и сколько в ней НДС?"}'
```

```json
{
  "answer": "Общая стоимость товара составляет 12 480 000 рублей, в том числе НДС 20% — 2 080 000 рублей.",
  "found": true,
  "citations": [
    {
      "text": "2.1. Общая стоимость Товара по Договору составляет 12 480 000\n(двенадцать миллионов четыреста восемьдесят тысяч) рублей, в том числе\nНДС 20% — 2 080 000 рублей.",
      "page": 1,
      "page_to": 1,
      "section": "2. ЦЕНА И ПОРЯДОК РАСЧЁТОВ",
      "chunk_index": 0,
      "char_start": 605,
      "char_end": 765
    }
  ],
  "diagnostics": null
}
```

### 7. Ответа в документе нет: «Кто является генеральным директором Поставщика?»

Сведений о руководителе в договоре нет. Реранкер пропустил в контекст один слабый фрагмент (0,097 при пороге 0,05), LLM вызвана и ответила, что сведений нет: `found=false`, `citations=[]`.

```bash
curl -X POST "localhost:8000/api/documents/329ffaf9-8847-4596-8e2e-a2ca40ada226/ask?debug=true" \
  -H 'Content-Type: application/json' -d '{"question": "Кто является генеральным директором Поставщика?"}'
```

```json
{
  "answer": "В документе нет сведений, необходимых для ответа на этот вопрос.",
  "found": false,
  "citations": [],
  "diagnostics": {
    "dense_hits": 5,
    "sparse_hits": 4,
    "fused_candidates": 5,
    "reranked_candidates": 5,
    "context_fragments": 1,
    "used_in_answer": [],
    "llm_called": true,
    "reranked": [
      {
        "chunk_index": 0,
        "page_from": 1,
        "page_to": 1,
        "section": null,
        "rerank_score": 0.0967,
        "dense_rank": 5,
        "sparse_rank": 1,
        "used_in_context": true
      },
      {
        "chunk_index": 2,
        "page_from": 2,
        "page_to": 3,
        "section": "3. СРОКИ И ПОРЯДОК ПОСТАВКИ",
        "rerank_score": 0.0447,
        "dense_rank": 1,
        "sparse_rank": 4,
        "used_in_context": false
      },
      {
        "chunk_index": 1,
        "page_from": 1,
        "page_to": 2,
        "section": "2. ЦЕНА И ПОРЯДОК РАСЧЁТОВ",
        "rerank_score": 0.0431,
        "dense_rank": 4,
        "sparse_rank": 2,
        "used_in_context": false
      },
      {
        "chunk_index": 3,
        "page_from": 3,
        "page_to": 5,
        "section": "5. ПРИЁМКА ТОВАРА",
        "rerank_score": 0.0378,
        "dense_rank": 3,
        "sparse_rank": 3,
        "used_in_context": false
      },
      {
        "chunk_index": 4,
        "page_from": 5,
        "page_to": 5,
        "section": "8. РАЗРЕШЕНИЕ СПОРОВ",
        "rerank_score": 0.0329,
        "dense_rank": 2,
        "sparse_rank": null,
        "used_in_context": false
      }
    ]
  }
}
```

### Расширенная проверка: 30 вопросов

`examples/test_questions.json` — 30 вопросов с ожидаемыми ответами: точные обозначения, переформулировки, ответы из нескольких мест, вопросы без ответа, попытки обойти правила. Скрипт `uv run python examples/run_test_questions.py` сверяет `found`, страницы и разделы цитат и пишет отчёт в `examples/results/test_questions_report.json`.

Результат: **29 из 30**. Расхождение — вопрос «Какой размер неустойки за нарушение сроков приёмки?»: модель отвечает по существу верно («в документе не указано… указана неустойка за нарушение сроков поставки — 0,1%»), но ставит `found=true` со ссылкой на п. 4.2.1, а ожидался `found=false`.

## Что доработать для production

- **Хранилище файлов.** Вместо общего тома — S3/MinIO за тем же портом `FileStoragePort`, Загрузка напрямую в S3 по presigned URL, чтобы тело файла не шло через API.
- **Лимит размера на прокси.** `client_max_body_size` в nginx или ingress. Проверка `Content-Length` в API не срабатывает, если клиент шлёт тело чанками без этого заголовка: тогда multipart целиком попадает во временный файл и отклоняется только после приёма.
- **Отказоустойчивость задач.** Сейчас при потере сообщения задача так и останется `queued`. Нужен периодический reconcile зависших `queued`/`running` по `updated_at` и dead letter queue. Отдельная очередь и пул воркеров под тяжёлые документы.
- **Сканы и сложная вёрстка.** OCR (Tesseract / PaddleOCR) для PDF без текстового слоя. Учёт колонтитулов и многоколоночной вёрстки. Таблицы — через layout-модели (Docling, Unstructured). Для DOCX — точные страницы через конвертацию в PDF (LibreOffice headless).
- **Нарезка по структуре.** Резать по границам пунктов и разделов, а не только по длине. Добавлять заголовок раздела в текст фрагмента для эмбеддинга (contextual chunk headers).
- **Качество.** Набор эталонных вопросов и ответов и офлайн-оценка: recall@k по ветвям, faithfulness и answer relevancy (RAGAS / LLM-as-judge) при смене моделей, порогов и `CHUNK_SIZE`. Подбор `RERANK_MIN_SCORE` на этом наборе.
- **Безопасность и доступ.** Аутентификация (API-ключи или OAuth), изоляция документов по арендатору (`tenant_id` в payload и фильтре), лимиты запросов, антивирусная проверка загрузок, секреты в Vault или KMS.
- **Удаление документов** (`DELETE /documents/{id}`) и переиндексация при смене модели эмбеддингов: новая коллекция и alias-переключение в Qdrant.
- **Наблюдаемость.** Структурные логи с `request_id` и `doc_id`, метрики Prometheus (латентность этапов, ошибки внешних API, глубина очереди), трассировка LLM (Langfuse / OpenTelemetry), учёт стоимости запросов OpenRouter.
- **Устойчивость к провайдерам.** Circuit breaker, fallback-модели (OpenRouter `models`/`provider` routing), кэш эмбеддингов запросов.
- **Масштабирование.** Qdrant в кластере с репликацией и snapshot-бэкапами, Redis Sentinel или Cluster, горизонтальное масштабирование API и воркеров, гибридный поиск с квантизацией dense-векторов.
