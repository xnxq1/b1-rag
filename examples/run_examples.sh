#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/results"
mkdir -p "$OUT"

pretty() { python3 -m json.tool --no-ensure-ascii; }

echo "1. Загрузка документа"
curl -sS -X POST "$BASE_URL/api/documents" \
  -F "file=@$DIR/contract.pdf" -F "title=Договор поставки № 47/2025" | pretty | tee "$OUT/01_upload.json"
TASK_ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['task_id'])" "$OUT/01_upload.json")

echo "2. Ожидание индексации"
for _ in $(seq 1 120); do
  STATUS=$(curl -sS "$BASE_URL/api/tasks/$TASK_ID")
  STATE=$(echo "$STATUS" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  [[ "$STATE" == "done" || "$STATE" == "failed" ]] && break
  sleep 2
done
echo "$STATUS" | pretty | tee "$OUT/02_task_status.json"
[[ "$STATE" == "done" ]] || { echo "Индексация не удалась"; exit 1; }

echo "3. Список документов"
curl -sS "$BASE_URL/api/documents" | pretty | tee "$OUT/03_documents.json"

ask() {
  local name="$1" question="$2" query="${3:-}"
  echo "-> $question"
  curl -sS -X POST "$BASE_URL/api/documents/$TASK_ID/ask$query" -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys; print(json.dumps({'question': sys.argv[1]}, ensure_ascii=False))" "$question")" \
    | pretty | tee "$OUT/$name.json"
}

echo "4. Вопросы"
ask 04_penalty "Какой штраф за просрочку поставки?"
ask 05_exact_clause "Что сказано в пункте 4.2.2?" "?debug=true"
ask 06_exact_amount "Какой размер НДС, если общая стоимость 12 480 000 рублей?"
ask 07_not_found "Кто является генеральным директором Поставщика?" "?debug=true"
