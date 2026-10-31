#!/usr/bin/env bash
# Run the three matched experiments sequentially in one eight-GPU session.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_path="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
cd -- "$script_dir"

base_dir="$PWD/log/embedding_head/gpu8/seed_1337"
queue_dir="$base_dir/queue"
mkdir -p "$queue_dir"
exec 9> "$queue_dir/queue.lock"
if ! flock -n 9; then
  echo "The eight-GPU experiment queue is already running." >&2
  exit 1
fi

modes=(tied untied_clone untied_random)
for mode in "${modes[@]}"; do
  if [[ -e "$base_dir/$mode/run1" || -e "$queue_dir/$mode.console.log" ]]; then
    echo "Existing results for $mode; inspect them and resume explicitly in a new run directory." >&2
    exit 1
  fi
done

current_mode=preflight
write_status() {
  python3 - "$queue_dir/status.json" "$1" "$current_mode" "${2:-0}" <<'PY'
from pathlib import Path
import datetime
import json
import sys

path = Path(sys.argv[1])
record = {"status": sys.argv[2], "mode": sys.argv[3], "exit_code": int(sys.argv[4]),
          "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n")
temporary.replace(path)
PY
}
on_exit() {
  queue_exit_code=$?
  trap - EXIT
  if (( queue_exit_code != 0 )); then
    write_status failed "$queue_exit_code"
  fi
  exit "$queue_exit_code"
}
trap on_exit EXIT
write_status checking

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TIKTOKEN_CACHE_DIR="$PWD/log/tiktoken_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Check data and free devices without constructing a model or CUDA context.
python3 - <<'PY'
from pathlib import Path
import json
import subprocess

preparation = json.loads(Path("edu_fineweb10B/preparation.json").read_text())
if preparation["status"] != "complete":
    raise RuntimeError("FineWeb preparation is not complete")
for name in ("token_counts.pt", "token_groups.pt", "manifest.json"):
    if not (Path("log/embedding_head/analysis_data") / name).is_file():
        raise FileNotFoundError(name)
if not Path("hellaswag/hellaswag_val.jsonl").is_file():
    raise FileNotFoundError("HellaSwag validation cache is missing")
devices = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True)
if {int(index) for index in devices.split()} != set(range(8)):
    raise RuntimeError("This queue expects GPUs 0 through 7")
processes = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
if processes.strip():
    raise RuntimeError("GPU compute processes are still present; release the old tasks first")
PY

cp -- "$script_path" "$queue_dir/launcher.sh"
for current_mode in "${modes[@]}"; do
  run_dir="$base_dir/$current_mode/run1"
  write_status running
  echo "Starting $current_mode on GPUs $CUDA_VISIBLE_DEVICES"
  python3 -u -m torch.distributed.run --standalone --nproc_per_node=8 \
    train_experiment.py --mode "$current_mode" --seed 1337 \
    --batch-size 16 --sequence-length 1024 --total-batch-tokens 524288 \
    --max-steps 19073 --warmup-steps 715 \
    --data-dir edu_fineweb10B \
    --groups log/embedding_head/analysis_data/token_groups.pt \
    --run-dir "$run_dir" \
    2>&1 | tee "$queue_dir/$current_mode.console.log"

  # Advance only after both the process and the final result confirm completion.
  python3 - "$run_dir/result.json" <<'PY'
from pathlib import Path
import json
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("status") != "complete" or result.get("completed_steps") != 19073:
    raise RuntimeError(f"Experiment did not finish the full training budget: {result}")
PY
  echo "Completed $current_mode"
done
current_mode=all
write_status complete
