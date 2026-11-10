# Task 5 steps (GR00T N1.7 on SimplerEnv Fractal). Sourced by groot/jobs/groot.pbs, which sets $G, $SRC, $PY_G, $HANDOVER and exports HF_TOKEN.
# Task names: s5__ckpt, s5__env, s5__data, s5__fp_<family>   (families: simpler/plan_task5.py)
ISAAC=$HANDOVER/runs/toolkits/Isaac-GR00T
S5=$G/s5
SIMPLER_PY=$ISAAC/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python

t_s5() {   # t_s5 <step>
  local step=$1
  mkdir -p "$S5"
  case "$step" in
    ckpt)   # official fine-tuned Fractal checkpoint without the DeepSpeed optimizer shards (about 28 GB, not needed for inference; decision G12)
      "$PY_G" - "$S5/checkpoints/fractal" <<'PY' || return 1
import sys, os
from huggingface_hub import snapshot_download
p = snapshot_download('nvidia/GR00T-N1.7-SimplerEnv-Fractal', local_dir=sys.argv[1], ignore_patterns=['global_step*'], token=os.environ['HF_TOKEN'])
print('checkpoint in', p)
PY
      du -sh "$S5/checkpoints/fractal" ;;
    env)    # one-time SimplerEnv environment of the Isaac-GR00T repo (python 3.10 island with ManiSkill2_real2sim), then its own import check
      # the official setup does not install msgpack-numpy, which the rollout client (gr00t/policy/server_client.py) imports (FP runs crashed on 2 Nov)
      cd "$ISAAC" && bash gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh \
        && VIRTUAL_ENV=$ISAAC/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv ~/.local/bin/uv pip install msgpack-numpy imageio-ffmpeg \
        && mkdir -p "$HANDOVER/runs/toolkits/bin" \
        && ln -sf "$("$SIMPLER_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$HANDOVER/runs/toolkits/bin/ffmpeg" \
        && touch "$S5/env_ok" ;;
    data)   # the calibration and held-out episodes of the manifests (data/fractal_*_manifest.json) from the fine-tuning dataset
      "$PY_G" - "$S5/data" "$SRC/data" <<'PY' || return 1
import sys, os, json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
out, man = Path(sys.argv[1]), Path(sys.argv[2])
repo, tok = 'IPEC-COMMUNITY/fractal20220817_data_lerobot', os.environ['HF_TOKEN']
(out / 'revision.txt').write_text(HfApi(token=tok).dataset_info(repo).sha)
n = 0
for name in ('calibration', 'heldout'):
    for e in json.loads((man / f'fractal_{name}_manifest.json').read_text())['episodes']:
        i = e['episode_index']; c = i // 1000
        for f in (f'data/chunk-{c:03d}/episode_{i:06d}.parquet', f'videos/chunk-{c:03d}/observation.images.image/episode_{i:06d}.mp4'):
            hf_hub_download(repo, f, repo_type='dataset', token=tok, local_dir=out); n += 1
print('downloaded', n, 'files')
(out / 'done').write_text('ok')
PY
      ;;
    fp_*)   # FP policy on one Google Robot task: official server + rollout client, 100 episodes, fixed seed 7 (log keeps the success rate)
      local fam=${step#fp_} env_name port log; env_name=simpler_env_google/google_robot_$fam
      port=$((20000 + ${PBS_JOBID%%.*} % 20000)); log=$S5/eval/fp/$fam.log; mkdir -p "$S5/eval/fp"
      [[ -x $SIMPLER_PY ]] || { echo "SimplerEnv environment missing (s5__env)" >&2; return 1; }
      ( cd "$ISAAC" && exec "$PY_G" gr00t/eval/run_gr00t_server.py --model-path "$S5/checkpoints/fractal" --embodiment-tag SIMPLER_ENV_GOOGLE \
          --use-sim-policy-wrapper --port "$port" > "$S5/eval/fp/$fam.server.log" 2>&1 ) &
      local spid=$! t=0
      until grep -q -i "ready\|listening\|serving" "$S5/eval/fp/$fam.server.log" 2>/dev/null; do
        kill -0 $spid 2>/dev/null || { echo "server died" >&2; return 1; }
        sleep 10; t=$((t + 10)); [[ $t -gt 900 ]] && { kill $spid; echo "server start timeout" >&2; return 1; }
      done
      # the rollout always records videos (default /tmp) and calls `ffmpeg`, which the nodes lack (crash of 4 Nov): use the static
      # ffmpeg of the venv's imageio-ffmpeg (symlink in runs/toolkits/bin) and keep the videos on scratch for Task 6
      mkdir -p "$S5/eval/fp/videos/$fam"
      ( cd "$ISAAC" && PATH=$HANDOVER/runs/toolkits/bin:$PATH "$SIMPLER_PY" gr00t/eval/rollout_policy.py --n-episodes 100 --policy-client-host 127.0.0.1 --policy-client-port "$port" \
          --max-episode-steps 300 --env-name "$env_name" --n-action-steps 1 --n-envs 5 --seed 7 --video-dir "$S5/eval/fp/videos/$fam" ) > "$log" 2>&1
      local rc=$?; kill $spid 2>/dev/null; wait $spid 2>/dev/null; return $rc ;;
    fa)   # rigour R5: flash-attn 2.8.3 built from source for torch 2.9 / sm_120 (no matching wheel) into its own folder; the server reads it only when FA_SITE is set
      local fa=$HANDOVER/runs/envs/flash-attn
      module load cuda/12.9 2>/dev/null || source /etc/profile.d/modules.sh && module load cuda/12.9
      rm -rf "$fa.partial"; mkdir -p "$fa.partial" /scratch/d250009/scratch_tmp
      CC=/usr/local/gcc-12.3/bin/gcc CXX=/usr/local/gcc-12.3/bin/g++ LD_LIBRARY_PATH="/usr/local/gcc-12.3/lib64:${LD_LIBRARY_PATH:-}" \
        CPATH=/scratch/d250009/uv-python/cpython-3.12.15-linux-x86_64-gnu/include/python3.12 \
        TORCH_CUDA_ARCH_LIST=12.0 FLASH_ATTN_CUDA_ARCHS=120 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 TMPDIR=/scratch/d250009/scratch_tmp PIP_CACHE_DIR=/scratch/d250009/pip-cache \
        UV_CACHE_DIR=/scratch/d250009/uv-cache ~/.local/bin/uv pip install --python "$PY_G" --no-build-isolation --no-deps --target "$fa.partial" flash-attn==2.8.3 || return 1   # the groot env has no pip; CPATH: /usr/include/python3.12 has no Python.h, so the headers come from a uv cpython 3.12.15 (same patch as the env) (7 Nov)
      PYTHONPATH="$fa.partial" "$PY_G" - <<'PY' || return 1
import torch, torch.nn.functional as F
from flash_attn import flash_attn_func
q, k, v = (torch.randn(2, 128, 16, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3))
o = flash_attn_func(q, k, v)
r = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
d = (o.float() - r.float()).abs().max().item(); print('flash-attn vs sdpa max abs', d); assert d < 2e-2
PY
      rm -rf "$fa"; mv "$fa.partial" "$fa" ;;
    fpfa_*)   # rigour R5: FP again with flash-attn in the server (the only setting left that differs from the official command), seed 7
      FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/checkpoints/fractal" "${step#fpfa_}" "$S5/eval/fp_fpfa" 7 ;;
    fprr_*|fpseed*_*)   # FP again for the open_drawer / close_drawer gap to the model card: fprr_<fam> = same seed 7 (noise of the run), fpseed<N>_<fam> = seed N
      local tag=${step%%_*} fam2 sd=7; fam2=${step#*_}; [[ $tag == fpseed* ]] && sd=${tag#fpseed}
      s5_run "$S5/checkpoints/fractal" "$fam2" "$S5/eval/fp_$tag" "$sd" ;;
    ckptpre)   # X9 (8 Nov): the Fractal checkpoint before NVIDIA's "Fix normalization" commit (b86e19b, 17 Mar 2026), for the open_drawer gap
      "$PY_G" - "$S5/checkpoints/fractal_b86e19b" <<'PY' || return 1
import sys, os
from huggingface_hub import snapshot_download
p = snapshot_download('nvidia/GR00T-N1.7-SimplerEnv-Fractal', revision='b86e19b', local_dir=sys.argv[1], ignore_patterns=['global_step*'], token=os.environ['HF_TOKEN'])
print('checkpoint in', p)
PY
      du -sh "$S5/checkpoints/fractal_b86e19b" ;;
    fppre*_*)   # X9: the pre-fix checkpoint with NVIDIA's exact command (flash-attn, no seed); fppre<k>_<fam> = repeat k
      local tag=${step%%_*}; FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/checkpoints/fractal_b86e19b" "${step#*_}" "$S5/eval/fp_$tag" none ;;
    fpof*_*)   # X7 (8 Nov): Fractal FP with exactly NVIDIA's command (flash-attn, no --seed); fpof<k>_<fam> = repeat k
      local tag=${step%%_*}; FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/checkpoints/fractal" "${step#*_}" "$S5/eval/fp_$tag" none ;;
    fp1e*_*)   # X17 (GOAL v8 G1): NVIDIA's command with one env (no seed): does one env replay a fixed configuration sequence (seed 2022 default)?
      local tag=${step%%_*}; S5_NENVS=1 FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/checkpoints/fractal" "${step#*_}" "$S5/eval/fp_$tag" none ;;
    sh95mk)   # W6 (9 Nov): Fractal checkpoint with 0.95 W on the 432 linear weights the 2-bit rows quantize (same names as Bridge), sharded safetensors
      "$PY_G" - "$S5/checkpoints/fractal" "$S5/ckpt_sh95/fractal" "$G/s5bq/perturb/aqlm_w2_error.json" <<'PY' || return 1
import json, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
src, out, ej = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
keys = set(json.loads(ej.read_text())); part = out.with_name(out.name + '.partial'); shutil.rmtree(part, ignore_errors=True); part.mkdir(parents=True)
for f in src.iterdir():
    if f.suffix == '.safetensors' or f.name.startswith(('rng_state', 'global_step')) or f.name in ('scheduler.pt', 'zero_to_fp32.py', 'latest'):
        continue
    (shutil.copytree if f.is_dir() else shutil.copy2)(f, part / f.name)
done = 0
for f in sorted(src.glob('*.safetensors')):
    t = {}
    with safe_open(f, 'pt') as fs:
        meta = fs.metadata()
        for k in fs.keys():
            w = fs.get_tensor(k)
            if k in keys:
                w = (w.double() * 0.95).to(w.dtype); done += 1
            t[k] = w.contiguous()
    save_file(t, str(part / f.name), metadata=meta)
assert done == len(keys), (done, len(keys))
shutil.rmtree(out, ignore_errors=True); part.rename(out); print('scaled', done, 'tensors into', out)
PY
      ;;
    wf75mk)   # X28 (10 Nov): Fractal checkpoint moved 25% of the way back to the pretrained base (alpha 0.75, 432 weights), sharded; wise.py
      "$PY_G" "$SRC/groot/wise.py" --source "$S5/checkpoints/fractal" --base "$G/base/GR00T-N1.7-3B" --error-json "$G/s5bq/perturb/aqlm_w2_error.json" --alpha 0.75 --out "$S5/ckpt_wf75/fractal" ;;
    wf75_*)   # X28: wf75 on one Google Robot family, seed 7, 100 episodes (as fp_* and sh95_*)
      s5_run "$S5/ckpt_wf75/fractal" "${step#wf75_}" "$S5/eval/wf75" 7 ;;
    wf75big_*)   # X28: wf75, 500 seeded episodes, flash-attn (as fpbig and sh95big)
      S5_NEP=500 FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/ckpt_wf75/fractal" "${step#*_}" "$S5/eval/wf75_big" 7 ;;
    sh95_*)   # W6: 0.95 W on one Google Robot family, seed 7, 100 episodes (as fp_*, SDPA)
      s5_run "$S5/ckpt_sh95/fractal" "${step#sh95_}" "$S5/eval/sh95" 7 ;;
    sh95big_*)   # W6: 0.95 W, 500 seeded episodes, flash-attn (as fpbig)
      S5_NEP=500 FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/ckpt_sh95/fractal" "${step#*_}" "$S5/eval/sh95_big" 7 ;;
    fpbig_*)   # X17: 500 seeded episodes (seed 7, 5 envs, flash-attn): the mean over the 108 drawer configurations with a 2-point SE
      S5_NEP=500 FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/checkpoints/fractal" "${step#*_}" "$S5/eval/fp_fpbig" 7 ;;
    fpgv*_*)   # X18 (GOAL v8 G1): NVIDIA's command with the pinned client venv (gymnasium 0.29.1, opencv-python-headless 4.10 only); fpgv<k>_<fam> = repeat k
      local tag=${step%%_*}; SIMPLER_PY=/scratch/d250009/pi05-handover/runs/toolkits/simpler_uv_pinned/.venv/bin/python FA_SITE=$HANDOVER/runs/envs/flash-attn s5_run "$S5/checkpoints/fractal" "${step#*_}" "$S5/eval/fp_$tag" none ;;
    *) echo "unknown s5 step $step" >&2; return 2 ;;
  esac
}

s5_run() {   # s5_run <model dir> <family> <out dir> [seed, default 7; "none" = no --seed, as NVIDIA's README]: server + rollout client for one Google Robot family, 100 episodes, seed 7 (as fp_*); the log keeps the success list
  # S5_EMB / S5_ENV / S5_NAS override embodiment tag, env name and n-action-steps (GOAL v7 B1: NVIDIA's Bridge command, b5__of__*)
  local model=$1 fam=$2 out=$3 seed=${4:-7} env_name port log spid t=0 rc sarg; env_name=${S5_ENV:-simpler_env_google/google_robot_$fam}
  sarg=(--seed "$seed"); [[ $seed == none ]] && sarg=()
  port=$((20000 + ${PBS_JOBID%%.*} % 20000)); log=$out/$fam.log; mkdir -p "$out/videos/$fam"
  [[ -x $SIMPLER_PY ]] || { echo "SimplerEnv environment missing (s5__env)" >&2; return 1; }
  ( cd "$ISAAC" && [[ -n ${FA_SITE:-} ]] && export PYTHONPATH="$FA_SITE${PYTHONPATH:+:$PYTHONPATH}"; cd "$ISAAC" && exec "$PY_G" gr00t/eval/run_gr00t_server.py --model-path "$model" --embodiment-tag "${S5_EMB:-SIMPLER_ENV_GOOGLE}" \
      --use-sim-policy-wrapper --port "$port" > "$out/$fam.server.log" 2>&1 ) &
  spid=$!
  until grep -q -i "ready\|listening\|serving" "$out/$fam.server.log" 2>/dev/null; do
    kill -0 $spid 2>/dev/null || { echo "server died" >&2; return 1; }
    sleep 10; t=$((t + 10)); [[ $t -gt 900 ]] && { kill $spid; echo "server start timeout" >&2; return 1; }
  done
  ( cd "$ISAAC" && PATH=$HANDOVER/runs/toolkits/bin:$PATH "$SIMPLER_PY" gr00t/eval/rollout_policy.py --n-episodes "${S5_NEP:-100}" --policy-client-host 127.0.0.1 --policy-client-port "$port" \
      --max-episode-steps 300 --env-name "$env_name" --n-action-steps "${S5_NAS:-1}" --n-envs "${S5_NENVS:-5}" "${sarg[@]}" --video-dir "$out/videos/$fam" ) > "$log" 2>&1
  rc=$?; kill $spid 2>/dev/null; wait $spid 2>/dev/null
  grep -q "^success rate" "$log" || return 1
  return $rc
}

# SimplerEnv-Bridge (WidowX), protocol of groot/simpler/README.md (mentor, origin/main 3f03bde): seeded server simpler_server.py,
# eval_simpler_bridge.py --episodes 120 --seed 7 --n-action-steps 4, 300 steps, one env; one task per call, resumable.
# Tasks: b5__vk (one-minute Vulkan / SimplerEnv check), b5__ev__<row>__<task> (row fp: s5b/checkpoints/bridge @ 940134b).
t_b5() {   # t_b5 <task>
  local t=${1#b5__} step row task out model port spid w=0 rc
  step=${t%%__*}
  [[ -x $SIMPLER_PY ]] || { echo "SimplerEnv environment missing (s5__env)" >&2; return 1; }
  case "$step" in
    vk) ( cd "$ISAAC" && "$SIMPLER_PY" -c "
import sapien, simpler_env
env = simpler_env.make('widowx_spoon_on_towel'); env.reset(seed=0); print('vulkan ok', env.unwrapped.__class__.__name__)" ) ;;
    ev)
      t=${t#ev__}; row=${t%%__*}; task=${t#*__}
      out=$G/s5b/eval/$row; mkdir -p "$out"
      local seed=7 fa=; [[ $row =~ ^fps([0-9]+)$ ]] && seed=${BASH_REMATCH[1]}; [[ $row == fpfa ]] && fa=$HANDOVER/runs/envs/flash-attn
      [[ $row =~ ^fpfas([0-9]+)$ ]] && { seed=${BASH_REMATCH[1]}; fa=$HANDOVER/runs/envs/flash-attn; }   # X19 (GOAL v8 G3): seeded protocol with flash-attn (fpfas<seed>) or SDPA (fps<seed>), more seeds   # H7 (8 Nov): FP with eval seed 8 / FP with flash-attn in the seeded server (numerics only)
      local extra=() mrow=$row; [[ $row == *sl ]] && { mrow=${row%sl}; extra=(--step-log "$out/steps" --video-dir "$out/videos"); [[ $mrow == quip ]] && mrow=quip_w2; }
      [[ $row == fe* ]] && { mrow=fp; extra=(--fresh-env --step-log "$out/steps"); [[ $row == fent* ]] && extra+=(--no-env-termination); }   # X3 / A12 (9 Nov): fe<k> = FP, new env per episode, step log; fent<k> = the same with --no-env-termination
      [[ $row == *nt ]] && { mrow=${row%nt}; extra=(--step-log "$out/steps" --no-env-termination); [[ $mrow == quip ]] && mrow=quip_w2; }   # X3: fpnt / quipnt = episodes run to 300 steps after a success   # X2 (8 Nov): fpsl / quipsl = same seeded run with per-step logs and videos
      local sopts=()   # GOAL v8 W4 (X16): FP with a server-side probe, no weight change
      case $row in pbax95) sopts=(--action-scale 0.95) ;; pbaa95) sopts=(--action-scale-all 0.95) ;; pbn07) sopts=(--init-noise-scale 0.7) ;; pbn13) sopts=(--init-noise-scale 1.3) ;; pbn16) sopts=(--init-noise-scale 1.6) ;; pbn20) sopts=(--init-noise-scale 2.0) ;; pbn11) sopts=(--init-noise-scale 1.1) ;;
        pbd2) sopts=(--denoise-steps 2) ;; pbd8) sopts=(--denoise-steps 8) ;; pbt11) sopts=(--attn-temp 1.108) ;; pbtb11) sopts=(--attn-temp-backbone 1.108) ;;
        pbes) sopts=(--resid-in-scale 1.136 1.108) ;; pbesl) sopts=(--resid-in-scale 1.136 1.0) ;;   # X24: residual input scale = 1 / (writer shrink of sh95): LM 1/0.88, vision 1/0.9025; pbesl = language model only
        pbdb5|pbdb5s*) sopts=(--dup-batch 5); [[ $row =~ ^pbdb5s([0-9]+)$ ]] && seed=${BASH_REMATCH[1]} ;; esac   # X20 (G3): server batch of 5 copies, seeded protocol (seed 7, or pbdb5s<seed>)
      [[ $row == pb* ]] && { [[ ${#sopts[@]} -gt 0 ]] || { echo "unknown probe row $row" >&2; return 1; }; mrow=fp; }
      if [[ $mrow == fp || $mrow == fprr || $mrow =~ ^fps[0-9]+$ || $mrow == fpfa || $mrow =~ ^fpfas[0-9]+$ ]]; then model=$G/s5b/checkpoints/bridge; else model=$G/s5bq/ckpt/$mrow/bridge; fi   # rows: t_bq imp; fprr = FP again, same seed (H3: run-to-run noise)
      [[ -d $model ]] || { echo "no model $model" >&2; return 1; }
      nvidia-smi --query-gpu=name --format=csv,noheader > "$out/gpu.txt"
      port=$((20000 + ${PBS_JOBID%%.*} % 20000))
      ( cd "$ISAAC" && [[ -n $fa ]] && export PYTHONPATH="$fa${PYTHONPATH:+:$PYTHONPATH}"; cd "$ISAAC" && exec "$PY_G" "$SRC/groot/simpler/simpler_server.py" --model-path "$model" --embodiment-tag SIMPLER_ENV_WIDOWX --port "$port" "${sopts[@]}" > "$out/$task.server.log" 2>&1 ) &
      spid=$!
      until grep -q "^serving" "$out/$task.server.log" 2>/dev/null; do
        kill -0 $spid 2>/dev/null || { echo "server died" >&2; return 1; }
        sleep 10; w=$((w + 10)); [[ $w -gt 900 ]] && { kill $spid; echo "server start timeout" >&2; return 1; }
      done
      ( cd "$ISAAC" && PATH=$HANDOVER/runs/toolkits/bin:$PATH "$SIMPLER_PY" "$SRC/groot/simpler/eval_simpler_bridge.py" --tasks "$task" \
          --episodes 120 --seed "$seed" --n-action-steps 4 --max-episode-steps 300 --port "$port" --out "$out/$task.json" "${extra[@]}" ) >> "$out/$task.log" 2>&1
      rc=$?; kill $spid 2>/dev/null; wait $spid 2>/dev/null
      [[ $rc -eq 0 ]] && python3 -c "import json,sys; r=json.load(open('$out/$task.json')); sys.exit(0 if len(r) >= 120 else 1)" || return 1
      if [[ $row != fp* && $task == widowx_spoon_on_towel ]]; then   # sanity gate of a 2-bit row (FP 101 of 120): 5% or less stops the row and frees its checkpoint
        python3 -c "import json,sys; r=json.load(open('$out/$task.json')); sys.exit(0 if sum(bool(x['success']) for x in r) / len(r) > 0.05 else 1)" \
          || { echo "spoon_on_towel at 5% or less: row stopped by the gate" > "$G/stopped/b5_$row"; rm -rf "$G/s5bq/ckpt/$row"; }
      fi ;;
    of)   # GOAL v7 B1 (H6): NVIDIA's own Bridge command (run_gr00t_server --use-sim-policy-wrapper, rollout_policy, 100 episodes, 5 async envs, n-action-steps 4, no seed); b5__of__<row>__<fa|sdpa>__<task>__r<k>
      local attn rep; t=${t#of__}; row=${t%%__*}; t=${t#*__}; attn=${t%%__*}; t=${t#*__}; task=${t%%__*}; rep=${t#*__}
      out=$G/s5b/official/${row}_$attn/$rep; mkdir -p "$out"
      if [[ $row == fp ]]; then model=$G/s5b/checkpoints/bridge; else model=$G/s5bq/ckpt/$row/bridge; fi
      [[ -d $model ]] || { echo "no model $model" >&2; return 1; }
      nvidia-smi --query-gpu=name --format=csv,noheader > "$out/gpu.txt"
      if [[ $attn == fa* ]]; then export FA_SITE=$HANDOVER/runs/envs/flash-attn; else unset FA_SITE; fi
      local ne=5; [[ $attn == *1 ]] && ne=1   # X4 (8 Nov): fa1 = NVIDIA's command with one env instead of 5
      local oseed=none; [[ $attn == fasd ]] && oseed=$((100 + ${rep#r}))   # H9 (9 Nov): fasd = NVIDIA's command, 5 envs, --seed 100+k (rollout_policy seeds the first reset of env i with seed+i, so each run draws its own layout chain instead of the 2022 chain)
      S5_NENVS=$ne S5_EMB=SIMPLER_ENV_WIDOWX S5_ENV=simpler_env_widowx/$task S5_NAS=4 s5_run "$model" "$task" "$out" "$oseed" ;;
    *) echo "unknown b5 step $step" >&2; return 2 ;;
  esac
}
