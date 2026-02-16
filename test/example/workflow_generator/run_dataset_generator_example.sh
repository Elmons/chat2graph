#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Dataset synthesis runner (no CLI args needed)
#
# 用法：
#   1) 先在 test/example/workflow_generator/utils.py 里填好图数据库连接(DB_CONFIG)
#   2) 按需修改本脚本顶部的参数区
#   3) 直接运行：
#        bash test/example/workflow_generator/run_dataset_generator_example.sh
#      或者(给执行权限后)：
#        chmod +x test/example/workflow_generator/run_dataset_generator_example.sh
#        ./test/example/workflow_generator/run_dataset_generator_example.sh
#
# 说明：本脚本会把下面这些变量翻译成 dataset_generator_example.py 的 CLI 参数。
#       你不需要在命令行手动传参，只需要改这里。
# ============================================================================

# -------------------------- 参数区（只改这里） ------------------------------

# 任务描述：用于引导 LLM 生成什么类型的图数据库问题
TASK_DESC='你的主要职责是解决关于图数据库的各种问题，包括实体查询、多跳推理等等'

# 数据集名称（会参与输出目录命名）
DATASET_NAME='query_real'

# 目标生成条数（最终会强制 >= 1）
SIZE=100

# 超时秒数：
#   - 0 或负数：禁用超时保护（可能卡很久，取决于模型/数据库）
#   - >0：开启超时保护；脚本内部会强制至少 60 秒
TIMEOUT_SECONDS=0

# 子图采样限制：影响每次抽取的子图规模，进而影响生成问题的复杂度/多样性/耗时
MAX_DEPTH=5   # 最长路径深度（>=1）
MAX_NODES=10   # 子图最大节点数（>=1）
MAX_EDGES=30  # 子图最大边数（>=1）

# 每个子图希望生成多少条样本（>=1）
NUMS_PER_SUBGRAPH=10

# 输出目录（每次运行会在该目录下创建一个 run_dir 子目录）
OUTPUT_DIR='test/example/workflow_generator/generated_datasets'

# 运行标签：用于区分不同实验；为空则不追加
RUN_TAG='exp1_100'  # 例如：'exp1'。留空 '' 表示不传

# 是否把运行时 stdout/stderr 也写入 run.log（同时仍会打印到控制台）
LOG_TO_FILE=true  # true/false

# （可选）打开 Dataset Synthesis debug 输出（依赖 app.core.common.system_env.SystemEnv）
# export DATASET_SYNTHESIS_DEBUG=1
# export DATASET_SYNTHESIS_DEBUG_MAX_CHARS=2000

# ------------------------ 参数区结束（下面一般不用改） ------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/dataset_generator_example.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[runner] ERROR: cannot find ${PY_SCRIPT}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "[runner] repo_root=${ROOT_DIR}"
echo "[runner] script=${PY_SCRIPT}"

# Prefer poetry if available (repo has pyproject.toml)
PY_RUN=(python)
if command -v poetry >/dev/null 2>&1 && [[ -f "${ROOT_DIR}/pyproject.toml" ]]; then
  PY_RUN=(poetry run python)
fi

declare -a ARGS
ARGS+=("--task-desc" "${TASK_DESC}")
ARGS+=("--dataset-name" "${DATASET_NAME}")
ARGS+=("--size" "${SIZE}")
ARGS+=("--timeout-seconds" "${TIMEOUT_SECONDS}")
ARGS+=("--max-depth" "${MAX_DEPTH}")
ARGS+=("--max-nodes" "${MAX_NODES}")
ARGS+=("--max-edges" "${MAX_EDGES}")
ARGS+=("--nums-per-subgraph" "${NUMS_PER_SUBGRAPH}")
ARGS+=("--output-dir" "${OUTPUT_DIR}")

if [[ -n "${RUN_TAG}" ]]; then
  ARGS+=("--run-tag" "${RUN_TAG}")
fi

# argparse.BooleanOptionalAction:
#   --log-to-file / --no-log-to-file
if [[ "${LOG_TO_FILE}" == "true" ]]; then
  ARGS+=("--log-to-file")
else
  ARGS+=("--no-log-to-file")
fi

echo "[runner] running: ${PY_RUN[*]} ${PY_SCRIPT} ${ARGS[*]}"

# 提示：utils.py 里 DB_CONFIG 默认是占位符，没填会直接 exit(1)
"${PY_RUN[@]}" "${PY_SCRIPT}" "${ARGS[@]}"

# 输出结构说明：
#   OUTPUT_DIR/<timestamp>_<dataset_name>_<size>[_<run_tag>]/
#     - dataset.json                生成的数据集(数组，每条是 row.model_dump())
#     - quality_audit.json          质量检查统计与分布
#     - reject_review.json          拒绝样本的原因统计/样例
#     - decision_review.json        决策/重写/挽救过程统计
#     - meta.json                   本次运行元信息（含路径、qa stats、是否通过等）
#     - run.log                     (如果开启 LOG_TO_FILE)
#     - progress_*.json / *.jsonl   过程追踪输出（候选、决策、事件、拒绝等）
