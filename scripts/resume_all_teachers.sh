#!/usr/bin/env bash
# Resume per-teacher batch grading. Each row below is:
#   <teacher-config>::<output-dir>
# The script preserves whatever ratings already exist in each output dir
# (rating_key is now correctly keyed on thema|quelle|teacher_name) and only
# calls the model for missing (topic, source, teacher) tuples in batches of
# up to 13 essays per topic — i.e. at most 15 calls per teacher.
#
# Run from the project root:
#   bash scripts/resume_all_teachers.sh
#
# Re-runs are idempotent: completed rows are skipped, failed rows are retried.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="outputs/full-run-generation/essays-wide.csv"
CFG_DIR="outputs/full-run-ratings-teacher-configs"
OUT_BASE="outputs/full-run-ratings-by-teacher"

# teacher-config-stem :: output-folder
PAIRS=(
  "claudeopus4-7-evaluator::claudeopus4-7-evaluator"
  "deepseek-v3-2-evaluator::deepseek-v3-2-evaluator"
  "gemini3-1pro-evaluator::gemini3-1pro-evaluator"
  "gpt-5-4-evaluator::gpt-5-4-evaluator"      # already has 194 rows merged from smoke-gpt-batch
  "kimi-k2-5-evaluator::kimi-k2-5-evaluator"
  "minimax-m2-7-evaluator::minimax-m2-7-evaluator"
  "qwen3-5-397b-a17b-evaluator::qwen3-5-397b-a17b-evaluator"
  "grok-4-1-fast-evaluator::grok-4-1-fast-evaluator"
  "doubao-seed-2-0-pro-evaluator::doubao-seed-2-0-pro-evaluator"
)
# glm-5-1-evaluator removed 2026-04-27: GLM API too slow, dropped as a teacher.
# minimax/qwen switched 2026-04-27 to router.shengsuanyun.com (highspeed/flash variants); old paratera data archived under .old-20260427.

PYTHON="${PYTHON:-python3}"

for entry in "${PAIRS[@]}"; do
  cfg="${entry%%::*}"
  out="${entry##*::}"
  cfg_path="${CFG_DIR}/${cfg}.json"
  out_path="${OUT_BASE}/${out}"
  log="${OUT_BASE}/${out}.stdout.log"
  err="${OUT_BASE}/${out}.stderr.log"

  echo "==========================================================="
  echo "[`date '+%F %T'`] Teacher: ${cfg} -> ${out_path}"
  echo "==========================================================="

  "${PYTHON}" scripts/batch_grade_essays.py \
    --dataset "${DATASET}" \
    --teachers "${cfg_path}" \
    --output-dir "${out_path}" \
    --include-student \
    --retry-errors \
    >>"${log}" 2>>"${err}" || {
      echo "  ! teacher ${cfg} failed; see ${err}" >&2
      continue
    }

  echo "  done. ratings rows: $(($(wc -l <"${out_path}/ratings_paper_aligned.csv") - 1))"
done

echo "All teachers processed."
