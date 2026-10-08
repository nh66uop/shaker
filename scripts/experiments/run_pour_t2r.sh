#!/usr/bin/env bash
# pour_fabric(text2reward) 학습 기동 — 서버/로컬 공용.
#   scripts/experiments/run_pour_t2r.sh <label> <iter_dir> [--num_envs N] [추가 train 인자...]
#   예) CUDA_VISIBLE_DEVICES=1 scripts/experiments/run_pour_t2r.sh t2r_i00 reward_gen/pour_bi/iter_00 --num_envs 256
# ★reward_code_path 는 절대경로로 넘긴다(hydra 는 env.* 필드를 덮고 env 가 런타임에 읽는다).
set -euo pipefail
LABEL="${1:?label}"; ITER="${2:?iter_dir}"; shift 2
HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # scripts/experiments/ → hdgp 루트
CODE="$(cd "$HDGP_ROOT" && realpath "$ITER/compute_reward.py")"
[ -f "$CODE" ] || { echo "no $CODE"; exit 1; }
python3 - "$ITER" <<'EOF'
import json, sys, pathlib
v = json.load(open(pathlib.Path(sys.argv[1]) / "validation.json"))
assert v["ok"], f"validation FAIL: {v['errors']}"
EOF
echo "reward: $CODE"
# 09.27 사용자 결정: LSTM + horizon 16 을 쓸 수 있게 과제·horizon·minibatch 를 환경변수로 받는다.
#   T2R_TASK      기본 open-short_b_pour_fab (MLP). LSTM 은 open-short_b_pour_fab-lstm
#   T2R_HORIZON   지정하면 horizon_length·seq_length 를 같은 값으로 덮는다(LSTM 은 seq 가 horizon 을 나눠야 한다)
#   T2R_MINIBATCH 지정하면 그 값, 아니면 num_envs×horizon/8 (최소 8192)
#   T2R_SAVE_FREQ 지정하면 save_frequency 를 덮는다(LSTM cfg 기본 500 은 체크포인트 고르기에 너무 성기다)
# minibatch 는 actor·central_value 둘 다 같은 값이어야 한다.
#   09.13 사용자 지시: 4096 env 로 확대(분산 축소). 8192 로 두면 32 minibatch 가 되어 느리다.
TASK="${T2R_TASK:-open-short_b_pour_fab}"
H="${T2R_HORIZON:-64}"
NE=""; prev=""; for a in "$@"; do [ "$prev" = "--num_envs" ] && NE="$a"; prev="$a"; done
MB=8192; if [ -n "$NE" ]; then MB=$(( NE * H / 8 )); [ "$MB" -lt 8192 ] && MB=8192; fi
[ -n "${T2R_MINIBATCH:-}" ] && MB="$T2R_MINIBATCH"
EXTRA=()
if [ -n "${T2R_HORIZON:-}" ]; then
  EXTRA+=("agent.params.config.horizon_length=$H")
  [[ "$TASK" == *-lstm ]] && EXTRA+=("agent.params.config.seq_length=$H")
fi
[ -n "${T2R_SAVE_FREQ:-}" ] && EXTRA+=("agent.params.config.save_frequency=$T2R_SAVE_FREQ")
# T2R_HYDRA_EXTRA: 공백으로 나눈 hydra override 를 그대로 덧붙인다(예: LSTM 학습률을 MLP 와 맞추기)
if [ -n "${T2R_HYDRA_EXTRA:-}" ]; then read -r -a _X <<< "$T2R_HYDRA_EXTRA"; EXTRA+=("${_X[@]}"); fi
echo "task=$TASK num_envs=${NE:-cfg} horizon=$H minibatch=$MB extra=${EXTRA[*]:-none}"
NOTE="${NOTE:-t2r $ITER}" exec "$HDGP_ROOT/train.sh" "$TASK" "$LABEL" \
    --headless "env.reward_code_path=$CODE" \
    "agent.params.config.minibatch_size=$MB" \
    "agent.params.config.central_value_config.minibatch_size=$MB" "${EXTRA[@]}" "$@"
