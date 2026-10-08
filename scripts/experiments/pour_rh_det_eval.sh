#!/usr/bin/env bash
# pour_fabric_mimic 결정론 자동 평가 — 학습 곡선(탐색 잡음 + 전 스텝 평균)과 실제 동작의 괴리를 없애기 위해,
# 새 체크포인트가 EVERY epoch 마다 나올 때 결정론 정책으로 64 env 를 돌려 에피소드 단위 지표(ep/*)를 기록한다.
#   ./scripts/experiments/pour_rh_det_eval.sh <run_label> <iter_dir> [EVERY=100]
#   출력: log/pour_fabric_mimic/<label>_deteval.tsv (epoch · 체크포인트 · ep/* 평균)
# 09.25 사용자 지시: "학습 로그와 실제 결정론 로그 차이가 너무 커서 분석이 어렵다 — 제대로 수정해서 학습 방향(기울기 등) 점검".
set -uo pipefail
LABEL="${1:?label}"; ITER="${2:?iter_dir}"; EVERY="${3:-100}"
HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NN="$HDGP_ROOT/log/rl_games/open-rh/both/pour-fab-mimic/$LABEL/nn"
RW="$HDGP_ROOT/$ITER/compute_reward.py"
OUT="$HDGP_ROOT/log/pour_fabric_mimic/${LABEL}_deteval.tsv"
TMP="$HDGP_ROOT/log/pour_fabric_mimic/${LABEL}_deteval_last.log"
[ -f "$OUT" ] || echo -e "epoch\tckpt\tn_done\ttilt_max\ttoward_max\tdir_err_at_max\tlift_max\tlipdz_at_max\tin_target_max\tin_source_min\tspill_end" > "$OUT"
while true; do
  for f in $(ls "$NN" 2>/dev/null | grep -E '^last_.*_ep_[0-9]+_' | sed -E 's/.*_ep_([0-9]+)_.*/\1 &/' | sort -n | cut -d' ' -f2); do
    ep=$(echo "$f" | sed -E 's/.*_ep_([0-9]+)_.*/\1/')
    [ $((ep % EVERY)) -eq 0 ] || continue
    grep -q "^${ep}	" "$OUT" && continue
    echo "[deteval] $(date +%T) ep$ep 평가 시작" >&2
    "$HOME/rl_ws/IsaacLab/isaaclab.sh" -p "$HDGP_ROOT/scripts/reinforcement_learning/rl_games/play.py" \
      --task open-rh_b_pour_fab_mimic --headless --num_envs 64 --eval_episodes 96 --dump_extras "ep/" \
      --checkpoint "$NN/$f" "env.reward_code_path=$RW" > "$TMP" 2>&1
    line=$(grep '^\[EXTRAS\]' "$TMP" | tail -1)
    if [ -z "$line" ]; then echo -e "${ep}\t${f}\tFAIL" >> "$OUT"; continue; fi
    get(){ echo "$line" | grep -oE "ep/$1=[-0-9.na]+" | cut -d= -f2; }
    echo -e "${ep}\t${f}\t$(get n_done)\t$(get tilt_max)\t$(get toward_max)\t$(get dir_err_at_max)\t$(get lift_max)\t$(get lipdz_at_max)\t$(get in_target_max)\t$(get in_source_min)\t$(get spill_end)" >> "$OUT"
    echo "[deteval] $(date +%T) ep$ep 기록" >&2
  done
  sleep 120
done
