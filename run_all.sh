#!/bin/bash
# Full-shot end-to-end: all 7 conditions (GPU), then the offline suite.
# One process per condition (module isolation between DUET/HAMT/RecBERT).
cd "$(dirname "$0")"
PY=/home/vfeliren1/pr65_scratch2/vfvic1/conda/envs/vln_duet_conformal/bin/python
mkdir -p logs

run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name ==="
  if $PY conformal_vln.py run "$@" > "logs/${name}.log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] DONE $name : $(grep -m1 'SANITY' logs/${name}.log || echo 'no sanity line')"
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $name (exit $?) -- tail:"
    tail -5 "logs/${name}.log"
  fi
}

run duet_full          --backend duet --action_space full
run duet_local         --backend duet --action_space local
run hamt               --backend hamt
run duet_full_reverie  --backend duet --dataset reverie
run hamt_reverie       --backend hamt --dataset reverie
run recbert_prevalent  --backend recbert --recbert_variant prevalent
run recbert_oscar      --backend recbert --recbert_variant oscar

echo "=== [$(date +%H:%M:%S)] GPU matrix done; offline suite ==="
for cmd in analyze dense transfer indist; do
  echo "--- $cmd ---"
  $PY conformal_vln.py $cmd 2>&1 | tail -12
done
$PY conformal_vln.py qualitative --condition duet_full 2>&1 | tail -2
$PY conformal_vln.py figures 2>&1 | tail -2
echo "--- verify ---"
$PY conformal_vln.py verify
echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
