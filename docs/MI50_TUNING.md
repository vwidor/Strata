# Dual-MI50 (`gfx906`) Tuning, Profiling & Benchmarks for Qwen3.8-Flash-Next IQ3_S

Measured engineering notes, GPU kernel profiles, engine changes, configuration recipes, rejected experiments, and next-step targets for running **Qwen3.8-Flash-Next GSQ-RCO IQ3_S** with **DFlash2** on dual **AMD Radeon Instinct MI50 32GB (`gfx906`)** GPUs.

---

## 1. Test Hardware & Toolchain

| Component | Spec |
|---|---|
| **GPUs** | 2× AMD Radeon Instinct MI50 32GB (`Vega 20` / `gfx906`, 60 CUs each, **wave64**, `1,024 GB/s` peak HBM2/card, `26.8 TFLOPS` FP16/card) |
| **Power Cap** | Stock **`180 W`** per GPU (all measurements taken at `180 W`) |
| **Inter-GPU Link** | `hipDeviceCanAccessPeer(0,1) = 0`, `(1,0) = 0` (KVM guest without PCIe P2P or XGMI bridge; peer transfers bounce through pinned host RAM) |
| **Host CPU / RAM** | 6× Intel Xeon E5-1650 v4 @ 3.60 GHz (AVX2, single NUMA node), `128 GiB` RAM, `63 GiB` `/dev/shm` tmpfs |
| **OS / Toolchain** | Debian 13 (Trixie), ROCm 5.7 (`libamdhip64 5.7`), `rocBLAS 5.5`, `clang++-17`, `STRATA_HIP_GFX906=ON`, `CMAKE_HIP_ARCHITECTURES=gfx906` |

### Interconnect Note (Layer-Split vs Tensor Parallelism)
- With a **2-stage pipeline layer-split**, inter-GPU transfers are tiny (`~160 KB/window` for hidden states + `~80 KB/slot` for DFlash2 taps), so lack of PCIe P2P costs `<1–2%`.
- **Tensor Parallelism (TP)** requires 2× AllReduce per layer (`96` AllReduces/token), which is severely penalized without direct P2P/XGMI.

---

## 2. Model Quantization Map & VRAM Layout

### Target Model: `Qwen3.8-Flash-Next GSQ-RCO IQ3_S`
- **Geometry:** 48 layers (**36 GDN** + **12 sparse QSA** with `top_k=2048`), 512 experts/layer, top-10 active (`50.29 GB` total experts; `~5.41 GB` active weights + `~0.25 GB` KV/state per token), PLE table (`320,001,536` rows, RAM-mapped), BF16 vision projector (`~956 MiB` VRAM, configured with `elastic: true` to unload when idle).
- **Exact GGUF tensor quantization breakdown:**
  - **LM Head (`output.weight`, `2560 × 248320`):** `Q6_K`
  - **Token Embedding (`token_embd.weight`):** `IQ4_XS` (`322 MiB` mapped host RAM)
  - **GDN dense projections:** `attn_qkv` (`Q6_K`), `attn_gate` (`Q4_K`), `ssm_out` (`Q8_0`)
  - **QSA dense projections:** `attn_q` (`IQ4_XS`), `attn_k` / `attn_v` (`Q6_K`), `attn_output` (`Q8_0`)
  - **Shared experts (`ffn_*_shexp`):** `Q4_K`, `IQ4_XS`, `Q8_0`
  - **Routed experts (`ffn_*_exps`):** `IQ3_XXS`, `IQ2_S`, `IQ3_S`, `IQ4_NL` (down projections), `Q5_K`
  - **Hyper-Connections & Router:** `BF16` weights / `F32` norms
- **DFlash2 drafter (`dflash2-original-q4.pack`):** 5 transformer layers (`hidden=2560`, Q4_0 = `625 MiB` VRAM + `212.9 MiB` pruned 106,299-token draft head), tapping layers `5, 15, 25, 35, 45` (3 on GPU 0, 2 on GPU 1 at `layer_split=26`). Each batch slot drafter (`STRATA_BATCH_DF2=1`) adds `~85 MiB` private K/V ring + landing buffer on GPU 1.

### Verified VRAM Layout (`layer_split = 26`, `--kv k8v4`, `vision.elastic = true`, `--batch 4`)
| GPU | Layers & Workloads | Resident Experts | VRAM Used / Total | Free VRAM | GTT |
|---|---|---|---:|---:|---:|
| **GPU 0** | Layers `0–25` (**100% resident: zero-doorbell graph**), 4 slot sessions (`2.54 GiB`), 2nd verifier + GDN snapshots (`284 MiB`), elastic vision slot | **`13,312 / 13,312` (100%)** (`24.19 GiB`) | `30,933 / 32,752 MiB` | **`1,818 MiB`** *(fits 956 MiB vision on demand)* | `14 MiB` |
| **GPU 1** | Layers `26–47`, LM head, solo DFlash2 + 4× `slot_df2` drafters *(optional `--vram-reserve-later-mib 10400` reserves headroom for a co-resident GPU 1 workload)* | **`7,363 / 11,264` (65.4%)** (`14.78 GiB`) | `30,107 / 32,752 MiB` | **`2,644 MiB`** | `303 MiB` |
| **Total** | All 48 layers | **`20,675 / 24,576` (84%)** (`97–100%` decode hit rate) | `61,040 / 65,504 MiB` | `4,462 MiB` | `317 MiB` |

*(If GPU 1 is dedicated exclusively to Strata with no co-resident workload, omit `--vram-reserve-later-mib 10400` so GPU 1 caches ~100% of layers 26–47 as well.)*

---

## 3. Engine Changes on This Branch (`mi50/qwen38-flash-next`)

1. **Adaptive Split-Window Pipelined DFlash2 (`4f27c18`, `engine/src/program/generate.cpp`):**
   - Enables `--pipeline-windows 2` with `--draft dflash2` (including `--draft-fallback none` with no MTP loaded) by registering DFlash2 taps on both even and odd verifiers (`ver`, `ver_b`, and later stages) and guarding MTP-only setup/polling behind `use_mtp`.
   - Uses async `df2.draft_launch(...)` / `cudaStreamQuery` polling (`pl_df2_taps` per parity) so DFlash2 drafting is hidden behind GPU 1 execution (`chain late = 0`).
   - **Adaptive Split-or-Full:** When `p_on >= theta` (default `0.20`), splits a DFlash2 draft into overlapped `A` (up to `STRATA_DF2_PW_HALF`, default `4` tokens) on GPU 1 and speculative `B` (up to `4` tokens) on GPU 0 using the existing GDN snapshot rollback mechanism; on low-confidence spans below `theta`, skips splitting and runs a full-width serial window (`up to --spec`) with zero rollback penalty.
2. **Pipelined DFlash2 Batch Drafting (`c9d1824`, `verify.hpp`, `verify.cpp`, `generate.cpp`):**
   - Adds `Verifier::gather_stage_taps(...)` copying a single stage's taps on `vk.stream()` (strictly hardware-ordered between consecutive group windows on that stage).
   - Copies Stage 0 taps onto each slot's private `slot_df2[b]->landing()` when Stage 0 completes a group (`pm_s0_ev[b]`), appends Stage 1 taps at verdict (`pm_copied[b]`), and launches async `slot_df2[b]->draft_launch(...)` (or `context_async(...)`) inside the 2-group pipelined `pmtp` batch loop (`--batch 4 --batch-groups 2`, `STRATA_BATCH_DF2=1`, `STRATA_PIPED_MTP=3` → up to `2 slots × 4 rows = 8 rows/group`).
   - Tracks `pl_last_par` so solo-to-batch `BADM` handoff always gathers taps from the active parity verifier.

---

## 4. Standardized Benchmark Progression

All numbers measured on the dual-MI50 180 W rig with `temperature=0.0`, `enable_thinking=False`, zero domain-specific tuning, and zero meaningful quality loss (`STRATA_ROUTE_TAIL_SKIP=0`). Batch metrics report **total throughput across all concurrent slots** (`per-slot` in parentheses).

| Stage | Solo Prefill *(2.3k fresh)* | Solo Decode Prose *(2.3k+128)* | Solo Decode Code *(256 tok)* | 3× Batch Total *(per-slot)* | 4× Batch Total *(per-slot)* | Reload Time |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline** (`split=24`, `int8`, `batch 3 / groups 3`, MTP) | `~350 tok/s` *(25–31k: `~527`)* | `25.1 tok/s` | `35.1 tok/s` | `~28 tok/s` (`~9/slot`) | `~28 tok/s` (`~7/slot`) | `~134 s` |
| **Kernel/Prefill/Batch/VRAM Tuning** (`gfx906` flags, `k8v4`, `split=25`, `groups=2`, `MTP_STAGE0`, `PIPED_MTP`) | `~500 tok/s` *(25–31k: `761–797`)* | `~33 tok/s` | `~44 tok/s` | `~65 pure` (`~22/slot`) | `~55 pure` (`~14/slot`) | `~134 s` |
| **Serial DFlash2** (`pw=0`, `--spec 6 --draft-conf 0.35`, `/dev/shm` arena) | `493.4 tok/s` | `42.2 tok/s` | `49.2 tok/s` | `39.1 pure` (`13.0/slot`, `33.8 e2e`) | `46.5 pure` (`11.6/slot`, `41.0 e2e`) | `19–21 s` |
| **Adaptive Split-Window Pipelined DFlash2** (`4f27c18`, `pw=2`, `split=26`, `--spec 8`, `half=4`) | `509.0 tok/s` | `39.6–42.1 tok/s` | `54.8–55.7 tok/s` | `33.0 e2e` *(undrafted)* | `45.2 e2e` *(undrafted)* | `21–22 s` |
| **Pipelined DFlash2 Solo + Batch** (`c9d1824`, `STRATA_BATCH_DF2=1`, `STRATA_PIPED_MTP=3`) | **`509.0 tok/s`** *(25–31k: `761–797`)* | **`38.5–42.1 tok/s`** | **`60.8 tok/s`** | **Prose: `64.5 rows/s`** (`61.3 e2e` / `~75 pure`, `25/slot`) · **Code: `64.7 e2e`** (`~78–80 pure`, `26/slot`) | **Prose: `73.1 rows/s`** (`69.9 e2e` / `~85 pure`, `21/slot`) · **Code: `86.9 rows/s`** (`82.2 e2e` / **`~102 pure`**, `25.5/slot`) | **`21–22 s`** |

### Live Zero-Restart `STRATA_PIPELINE_SWITCH` A/B Sweep (`split=26`, `--spec 8`, `--draft-conf 0.35`)
Measured on the same warmed expert cache without restarting:
- `pw=0` (serial baseline): Prose `31.2 tok/s`, Code `45.3 tok/s`.
- `pw=2 half=4 theta=0.20`: Prose **`39.6 tok/s`** (`+27%`), Code **`54.8 tok/s`** (`+21%`).
- `pw=2 half=5 theta=0.30`: Prose `37.8 tok/s`, Code **`55.7 tok/s`** (`+23%`).
- `pw=2 half=4 theta=0.55–0.75` / `hit=0.55`: Prose `35.3–37.2 tok/s`, Code `49.8–52.7 tok/s`.
- **Why low `theta` (`0.20`) wins on MI50:** GPU 0 (100% resident zero-doorbell) finishes speculative `B` (`~24–26 ms`) before GPU 1 finishes `A` (`~35–38 ms`), so rollbacks cost only `~1 ms` (GDN D2D restore + commit replay) while hits save `~26 ms` (break-even `p_on ≈ 4%`). Measured `p_on` deciles `0.7 / 0.8 / 0.9` on code achieved **`70% / 92% / 91%`** on-path hit rates.

---

## 5. Cycle-Accurate MI50 GPU Kernel Profile (`STRATA_VERIFY_PROFILE=1`)

Measured at `layer_split=26`, `--pcie-frac 0`:

| Kernel Stage | `T = 1` *(GPU 0 / GPU 1)* | Solo Code `avg T = 3.60` *(GPU 0 / GPU 1)* | Batch `T = 7..8` *(GPU 0 / GPU 1)* | Share of Batch GPU Compute |
|---|---|---|---|---:|
| **Dense GEMVs (`native_mmvq.cu`: `q8+qkv`, `ab`, `z`, `out-proj`, `q+q-idx`, `shared`)** | `4.77 / 4.24 ms` | `9.73 / 8.13 ms` | **`11.08 / 9.59 ms`** | **32.3%** |
| **LM Head (`output.weight`, `Q6_K`, `native_mmvq.cu`)** | `0.00 / 1.16 ms` | `0.00 / 2.32 ms` | **`0.00 / 3.79 ms`** | **5.9%** *(Total MMVQ: **38.2%**)* |
| **VRAM Hits (`iq_kernels.cu`: routed MoE experts)** | `2.89 / 2.46 ms` | `6.16 / 5.17 ms` | **`8.68 / 6.82 ms`** | **24.2%** |
| **Hyper-Connections (`hc0` + `hc1` norm/down/up, `BF16`)** | `3.58 / 2.86 ms` | `4.81 / 3.67 ms` | **`6.20 / 4.50 ms`** | **16.7%** |
| **`waitCPU` (CPU expert misses on 6-core Xeon)** | `0.00 / 0.10 ms` | `0.00 / 3.38 ms` | **`0.00 / 6.40 ms`** *(cold: `10.4 ms`)* | **17% of GPU 1** |
| **Router (`proj + topk`)** | `1.91 / 1.68 ms` | `1.22 / 1.05 ms` | **`2.37 / 2.02 ms`** | **6.9%** |
| **QSA Attention + Scores + Top-K** | `1.16 / 1.15 ms` | `0.65 / 0.68 ms` | `0.44 / 0.44 ms` *(short)* / `1.37 / 1.44 ms` *(long)* | **1.4–4.4%** |
| **Total GPU Window Time** | **`15.82 / 15.82 ms`** | **`23.66 / 27.37 ms`** | **`32.36 / 37.73 ms`** | — |

### Roofline Takeaways
- **`VRAM hits` (`iq_kernels.cu`):** Effective bandwidth is only **`175 GB/s`** at `T=1` and **`126 GB/s`** at `T=3.6` (`12–17%` of MI50's `1,024 GB/s` peak) — bound by dequant/sign-mask ALU and 32-lane logical warp emulation (`strata_hip.h`), not HBM2 bandwidth.
- **`Q6_K` LM Head & Dense GEMVs (`native_mmvq.cu`):** LM head drops from **`448 GB/s`** at `T=1` (`1.16 ms`) to **`137 GB/s`** at `T=7..8` (`3.79 ms`, `3.27×` slower for the exact same weights).

---

## 6. Tested & Rejected Experiments

| Experiment | Measured Result | Why Rejected |
|---|---|---|
| **`--pcie-frac 0.55` + `STRATA_PCIE_AUTO_SPLIT=1`** | Solo code `58.0` vs `60.8`; 4× code `81.9` vs `86.9`; 3× prose `58.2` vs `64.5` | At `99%+` steady-state hit rate, `~0.4` misses/layer are faster on CPU than PCIe DMA + staging kernel + `waitB` sync |
| **10-flag HIP decode/batch fusion bundle** (`STRATA_SIDE_MIXER`, `LFUSE`, `QFUSE`, `VERIFY_QDEDUP`, `FUSE_HITS`, `PLE_BATCH`, `DF_PLE`, `BATCH_SLOT_RUNS`, `DF2_PRIO`, `PIPED_LOOKUP`) | `-3.5%` to `-6.2%` across solo & batch (`58.7` vs `60.8` code; `81.5` vs `86.9` 4× code) | Likely HIP side-stream sync overhead (`SIDE_MIXER`) and wave32-tuned `LFUSE`/`QFUSE`; test HIP-safe flags individually instead |
| **Wave32 decode flags** (`STRATA_EXPERT_V2`, `EXPDEC`, `HC_Q8`, `GR_Q32_G=2`, `GR_Q8_PFS`, `GR_UP75`, `ATTN_LANECELL`, `ATTN_MERGE_V2`, `TSUM`, `GDN_SPLIT`) | `1`-tok decode regressed `31–32 ms → 35–37 ms` (`HC_Q8` also cost `0.32 GiB`/GPU) | Overrode `gfx906`'s native Mode 7 / `GR_SPLIT` kernels |
| **`STRATA_DF2_STAGE=first`** (DFlash2 on GPU 0) | **SEGFAULT (`exit -11`)** on first request | Tap-gathering crash when `P2P=0` and DFlash2 is on stage 0; default GPU 1 works cleanly |
| **Serial DFlash2 `--spec 8 --draft-conf 0.25`** (`pw=0`) | Prose `35.7` vs `42.2`, Code `44.8` vs `49.2` | Wider serial windows waste verify time on rejected tails (`--spec 8` wins only with `pw=2` split-window) |
| **`--prefill 4096` + `STRATA_PF_W16_CACHE_MIB=512`** | Used `~800 MiB` extra VRAM on GPU 1 (`256` was inert: `0 MiB`) | Reduced free VRAM headroom too far; kept `--prefill 2048` |
| **`STRATA_BATCH_SELF_COMMIT=1` + `STRATA_PIPED_MTP`** | Engine warned and disabled `STRATA_PIPED_MTP` | Mutually exclusive |
| **`mmvq_il_parity` on HIP (as-is)** | Throws `"native_quantize_q8_1_il is CUDA only"` | Currently behind `#if !defined(__HIPCC__)` in `native_mmvq.cu:4020` |

---

## 7. Recommended Dual-MI50 Production Profile (Anonymized)

Note: if using a custom `model_name` not matched by `drafting.py:supported_model()`, set `"split_engine_note": true` in the JSON profile so the launcher does not strip `--draft dflash2`.

```json
{
  "exe": "/path/to/build-906/strata",
  "backend": "hip",
  "gpu": [0, 1],
  "gpu_order": "as_given",
  "layer_split": "28",
  "parallel": 4,
  "drafting": "dflash2",
  "mtp": false,
  "remote_expert_opt": false,
  "split_skip_if_fits": false,
  "host": "127.0.0.1",
  "port": 11433,
  "env": {
    "STRATA_SPLIT_OVERLAP": "0",
    "STRATA_ROUTE_TAIL_SKIP": "0",
    "STRATA_BATCH_YIELD_CHUNKS": "1",
    "STRATA_HIP_PROMPT_F16": "1",
    "STRATA_PW_WITH_BATCH": "1",
    "STRATA_GFX906_ATTN_QUERY_SWIZZLE": "1",
    "STRATA_GFX906_ATTN_REDUCE12": "1",
    "STRATA_GR_SPLIT": "3",
    "STRATA_WAVE_MMVQ_V2": "1",
    "STRATA_EXP_MODE": "11",
    "STRATA_MMVF_WAVE": "1",
    "STRATA_HIP_ROUTER_WAVE_FAST": "1",
    "STRATA_HANDOFF_PIPE": "4",
    "STRATA_SLOT_COPY_DIRECT": "1",
    "STRATA_SLOT_COPY_BATCHSYNC": "1",
    "STRATA_QSA_SLOTS": "1",
    "STRATA_CKPT_POOL": "1",
    "STRATA_CKPT_THREADS": "1",
    "STRATA_PIPE_BYIELD_NODRAIN": "1",
    "STRATA_HIP_SWIGLU_FUSED": "1",
    "STRATA_SELECT_SIMT": "2",
    "STRATA_TOPK_REG_MIN_BLOCKS": "4096",
    "STRATA_GDN_HEAD": "1",
    "STRATA_GDN_PP": "2",
    "STRATA_GDN_CONVL2": "1",
    "STRATA_GDN_NOY": "1",
    "STRATA_FUSE_DECODE": "1",
    "STRATA_FUSE_HEAD_GR": "1",
    "STRATA_Q8_PACKED": "1",
    "STRATA_SMALL_UNR": "3",
    "STRATA_CVEC_FUSE": "1",
    "STRATA_PF_FUSED": "1",
    "STRATA_SLOT_READ": "1",
    "STRATA_SLOT_READ_WIDE": "1",
    "STRATA_SLOT_READ_AHEAD": "1",
    "STRATA_PIPE_ADAPT_EVERY": "auto",
    "STRATA_CRASH_BT": "1",
    "STRATA_BATCH_DF2": "1",
    "STRATA_PIPED_MTP": "3",
    "STRATA_BATCH_DF2_CAP": "1",
    "STRATA_DF2_Q4_NARROW": "1",
    "STRATA_PCIE_NEVER": "1",
    "STRATA_CPU_IQ_ALL_EXACT1": "1",
    "STRATA_SPLIT_OWN_BUFFERS": "0",
    "STRATA_PF_W16_CACHE_MIB": "0",
    "STRATA_DF2_HEAD": "indexed",
    "STRATA_TRIM_VERIFY_SCRATCH": "1",
    "STRATA_LEAN_SLOTS": "1",
    "STRATA_SHARED_KV_CELLS": "131072",
    "STRATA_PW_PENALTY": "1",
    "STRATA_PW_FORK_B": "1",
    "STRATA_DF2_PW_DYN": "1",
    "STRATA_BATCH_DYN_GROUPS": "1",
    "STRATA_BATCH_SLOT_RUNS": "1",
    "STRATA_FUSE_HITS": "1",
    "STRATA_ROCBLAS_TUNING": "/path/to/gfx906-rocblas-2.47.0.txt",
    "STRATA_I8_TM": "7",
    "STRATA_QSA_PA_GFX906": "1",
    "STRATA_PF_I8_GPU_PLAN": "1",
    "STRATA_ATTN_MERGE_V2": "1"
  },
  "args": [
    "--pack", "/path/to/pack-iq3s",
    "--native", "/path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "2048",
    "--prefill-pipe", "1",
    "--ple-io", "ram",
    "--pcie-frac", "0",
    "--trim-stage-weights",
    "--spec", "8",
    "--lookup-max-t", "8",
    "--pipeline-windows", "2",
    "--max-context", "102400",
    "--kv", "k8v4",
    "--vram-reserve-mib", "1024",
    "--batch-groups", "2",
    "--conversation-cache-mib", "16384",
    "--conversation-cache-slots", "16",
    "--conversation-cache-min-tokens", "4096",
    "--conversation-cache-park-gain", "512",
    "--shared-expert-arena", "/dev/shm/strata-iq3s.arena",
    "--draft", "dflash2",
    "--draft-model", "/path/to/dflash2-original-q4.pack",
    "--draft-fallback", "none",
    "--mtp-draft-vocab", "/path/to/draft_vocab.bin",
    "--draft-conf", "0.35"
  ],
  "vision": {
    "gpu": true,
    "cuda_device": 0,
    "elastic": true,
    "max_tokens": 1024
  },
  "split_engine_note": true
}
```

---

## 8. Prioritized Next Steps (Easiest / Highest-ROI First)

1. **Support `penalty_last_n > 0` in `pl_pw >= 2` (`generate.cpp`):** Requests with repetition/presence penalties currently print `--pipeline-windows: a request with repetition penalties (penalty_last_n) decodes serially` and fall back to `pw=0`. Since Stage 1 (`V1(A)`) executes strictly in-order, updating its penalty history from `consumed` (~20 lines C++) unlocks pipelined DFlash2 for penalized requests too.
2. **A/B safe HIP fusion flags individually (`0` compile):** Test `STRATA_BATCH_SLOT_RUNS=1` (fuses contiguous 4-row-per-slot QSA/indexer/attention launches in piped DFlash2 batch), `STRATA_FUSE_HITS=1`, and `STRATA_DF2_PRIO=1` one by one.
3. **Port `native_mmvq_il` to HIP + tune `gfx906` tables (`native_mmvq.cu`):** Lift the `#if !defined(__HIPCC__)` guard, verify bitwise parity with `mmvq_il_parity --bench --emit-table` on MI50, and add `kIlRows_gfx906` / `kIlRows58_gfx906`. Targets **38.2% of all GPU compute (`24.46 ms/win`)**.
4. **Native wave64 `gfx906` expert hit kernels (`iq_kernels.cu`, `strata_hip.h`):** Targets **24.2% of all GPU compute (`15.50 ms/win`, currently `126–175 GB/s`)** — eliminate 32-lane logical warp overhead in `Split<18/21/22>`, audit VGPR spills, and switch unsigned `strata_dp4a` to `__builtin_amdgcn_udot4`.
5. **Dynamic split sizing & token-conditioned `B` block in Pipelined DFlash2:** Pick the `A`/`B` split dynamically from `pp[0..6]` instead of fixed `half=4` and test conditioning DFlash2's block on `B.tok[1..B.T-1]`.
6. **Modern ROCm (6.3+ / 7.x) + Clang 20+ container rebuild for `gfx906`:** Upgrade past Debian Trixie ROCm 5.7 / Clang 17 / `rocBLAS 5.5` for better VGPR allocation and faster FP16 prefill GEMMs.

---

## 9. Static Code Audit & Projected Roadmap (unconfirmed, not yet measured)

Four read-only audits of `550a628` (MMVQ/router, MoE experts and CPU misses, pipeline/DFlash2, prefill/HC/GDN/QSA/build). Everything below comes from **reading code**. No build or GPU run backs these findings yet. Treat each as unconfirmed static analysis until it is A/B tested on the rig.

### 9.1 Flags that do nothing on this gfx906 build
- `STRATA_PF_FUSED=1`: the gfx906 `dev_info()` returns before setting `ok`, so `native_supported()` declines the IQ layers (`moe_fused_iq.cu`). Prefill already uses the grouped INT8 expert kernels (`native_expert_i8`).
- `STRATA_ROCBLAS_TUNING`: CMake leaves the rocBLAS solution-table hook and `tune_rocblas` out of the gfx906 compatibility build (`strata_gpu_engine.cmake`).
- `STRATA_GDN_CHUNKED=1`: the HIP chunked recurrence needs gfx11/gfx12 WMMA.
- `STRATA_GFX906_ATTN_QUERY_SWIZZLE` / `STRATA_GFX906_ATTN_REDUCE12`: both need KV mode 1 (INT8 K and V). The main model runs `k8v4` (mode 3), so they only reach an INT8 drafter pool.
- `STRATA_Q8_PACKED=1`: the feature is on, but no Q8_0 matrix in this model passes its size rule (`ssm_out`/`attn_output` are `6144 → 2560`, input above 4096; shared experts are 640 wide).
- `STRATA_TOPK_REG_MIN_BLOCKS`: only affects calls that pass an active block count. Ordinary decode calls do not.
- Forcing `STRATA_QIL=1` on HIP can call an interleave producer stub that throws.

### 9.2 Other findings
- **`chain late = 0` does not prove DFlash2 is hidden.** The counter only counts MTP chains (`chain_kind == 1`). DFlash2 uses `df2_kind`/`df2_live`. Trace draft-ready time against the verdict instead.
- **Batch slots run a 7-output draft head** but `STRATA_PIPED_MTP=3` checks at most 3 proposals. The Q4_0 drafter GEMMs on gfx906 (variant 17) compute 8 columns even for 1.
- **Empty PCIe branch at `--pcie-frac 0`.** The partial-resident graph still waits on B and launches Mode 7 PCIe CTAs that exit at their group guard.
- **Mode 7 IQ gate/up spends 16 dot ops per 32 values.** Half are sign corrections. The grid magnitudes are strictly positive (IQ3_XXS 4–62, IQ3_S 1–15, IQ2_S 8–43), so `(grid ^ mask) + (mask & 0x01010101)` makes the signed bytes exactly and cuts it to 8 dots.
- **Q5_K gate/up runs the CUDA-layout `native_gu_kernel<13>`** by default on HIP. `STRATA_KQ_SPLIT=1` selects the grouped `Split<13>` path.
- **Hot dots are already signed `sdot4`.** Porting the unsigned `__dp4a` overload to `udot4` is correct but saves about 0 ms on these kernels. This corrects item 4 of section 8.
- **The interleaved MMVQ path is compiled out on HIP.** A port should reuse the `Q81Il` layout but keep the gfx906 wave64 sum tree. `kIlRows*` and `STRATA_MMVQ_IL_*` have no effect today. This refines item 3 of section 8.
- **Router at T=1:** the single-token top-10 launches 256 threads, returns all but one logical half-wave, and still hits `__syncthreads()`. `route_multi` uses a wave barrier.
- **Hyper-connections:** the split down kernel's LDS tile is `T × 2,048 × 4` bytes, 64 KiB at T=8, so one block per CU. HC moves 13.19 MB per read (0.67/0.57 ms per window at full HBM speed) against 3.58/2.86 ms measured at T=1. `HC_PERSIST` rejects `warpSize != 32`.
- **`STRATA_DF2_STAGE=first` crash:** first suspect is the reverse-direction `cudaMemcpy2DAsync` in the prefill tap callback, done across devices while holding `dland.mu` with P2P off.
- **Penalized requests:** the `pl_pw >= 2` fallback checks `hist_n > 0`, so it also triggers for a nonzero window with neutral penalties.

### 9.3 Projected roadmap (estimates from the measured profile)
Model: each kernel step cuts a share of one bucket of the section 5 profile. Pipelined throughput follows the slower GPU (weight 0.7 solo code, 0.5 prose, 0.9 batch), and 80–85% of wall time is taken as GPU-bound. On GPU 1, half of each kernel saving is assumed to turn into longer CPU wait, scaled by the wait still left. Steps 3 and 10 use direct ranges from the pipeline audit. Values are the middle estimate. Ranges come from the audit low/high bounds. Baselines are the measured `c9d1824` numbers (prose uses the 40.3 midpoint).

| Step (cumulative tok/s after it, GPU 1 shared as measured) | Solo prose | Solo code | 3× batch prose | 4× batch code |
|---|---:|---:|---:|---:|
| Today (measured) | 38.5–42.1 | 60.8 | 61.3 | 82.2 |
| 1. Flag-only A/B round (`CPU_IQ_ALL_EXACT1`, `KQ_SPLIT`, `FUSE_HITS`, `BATCH_SLOT_RUNS`, `--pool-tasks`, `PIPE_ADAPT_EVERY`) | 40.7 | 61.4 | 62.3 | 83.5 |
| 2. Cut GPU 1 CPU-expert wait (zero-share PCIe graph, O(n) host plan, residency by wait saved per MiB) | 41.8 | 63.3 | 65.4 | 87.7 |
| 3. Lighter DFlash2 drafting (batch head 7→3, Q4_0 widths 1–4) | 42.8 | 64.9 | 68.6 | 93.4 |
| 4. Faster Q6_K LM head | 43.3 | 65.8 | 70.5 | 95.8 |
| 5. Wide dense GEMVs (activation reuse) | 44.4 | 67.9 | 73.0 | 99.3 |
| 6. Router top-10 + BF16 tiles | 44.6 | 68.2 | 73.6 | 100.2 |
| 7. MoE hit kernels (signed bytes, decode reuse, wave64) | 45.5 | 69.4 | 74.9 | 101.9 |
| 8. Hyper-connection rework (LDS tiles, exact-T, wave64 Q8) | 46.7 | 71.1 | 76.5 | 104.1 |
| 9. Batch GDN, QSA + leftovers | 47.4 | 72.0 | 77.7 | 105.7 |
| 10. Smarter split + batch scheduling | **50.0** | **77.8** | **81.6** | **110.4** |
| Range of the total | 44.9–56.0 | 68.9–88.5 | 71.3–94.5 | 96.5–127.9 |
| With GPU 1 dedicated (drop `--vram-reserve-later-mib 10400`): that step alone | 43.3 | 65.8 | 69.4 | 93.0 |
| With GPU 1 dedicated: total | **54.6** (48.8–61.2) | **86.4** (76.0–98.6) | **93.0** (81.1–107.1) | **125.9** (109.8–144.9) |

While GPU 1 still waits on CPU misses, faster GPU 1 kernels mostly turn into more wait: in the shared case GPU 1 stays the slower stage (about 30.1 vs 25.5 ms per batch window after step 10). With GPU 1 dedicated the two stages end near balance (24.4 vs 25.5 ms).

Prefill (assumed shares of prefill time, pending `STRATA_PREFILL_TIMING`): P1 INT8 tile sweep and device-built group records (~45%) → 559 fresh / 856 long; P2 rocBLAS tuning enabled for gfx906 (~18%) → 579 / 887; P3 GDN prompt recurrence (~12%) → 598 / 915; P4 QSA mode-3 FP16-pair attention (~8%) → 614 / 939; P5 chunk and copy scheduling (2–5%) → **636 fresh (590–688) / 973 long (903–1,053)** from 509 / 761–797.

Not counted: pipelining for penalized requests (same-cache `pw=2` gave +27% prose and +21% code), a ROCm 7.x rebuild (no controlled MI50 run), and moving the drafter to GPU 0 or split 27/28 (GPU 0 has 862 MiB beyond vision; the drafters need about 1.3 GiB and a layer about 953 MiB of experts).

### 9.4 Prepared for the next benchmark window (built, not yet measured)
New opt-in switches. The default path is unchanged, and none of them has a measurement yet:

| Switch | Roadmap step | What it does | Output |
|---|---|---|---|
| `STRATA_BATCH_DF2_CAP=1` | 3 | With `STRATA_PIPED_MTP=N`, the slot drafters make N drafts a window, not `--spec - 1` (head, top-k and selector walk for N rows; the 8-row block is unchanged). | The first N drafts are bitwise the same (`set_max_drafts`). |
| `STRATA_DF2_Q4_NARROW=1` | 3 | Q4_0 drafter GEMMs with 1–4 columns compute only those columns (variants 17, 26, 31, 36, 38). | Bitwise the same per column. |
| `STRATA_PCIE_NEVER=1` | 2 | With `--pcie-frac 0`, verify windows skip the PCIe flag wait and the empty PCIe grouped launches, and the pool is told never to plan a PCIe share (a request's own `pcie_frac` included). | Unchanged: the share was always empty. |
| (always on) | 2 | The host expert plan uses an id table and per-expert entry chains instead of O(n²) scans. | Identical groups and order (checked on 200,000 random windows). |

### 9.5 Measured: round A (2026-11-25)
Dual MI50, the production profile (`layer_split` 26, `--pipeline-windows 2`, `--batch 4`, `STRATA_BATCH_DF2=1`, `STRATA_PIPED_MTP=3`, `--vram-reserve-later-mib 16500`: GPU 1 keeps 16.5 GB free for a second model), the second model stopped. 3 runs per arm, each arm freshly loaded (a cold expert cache: 80-92% hits, against 94% in long production use), 256-token answers, thinking off. tok/s are medians; "paired" is the median ratio of the same prompt in the same run against `base` (the new build, switches off), with the share of pairs that were faster.

| Arm | Solo prose | Solo code | 3x prose | 3x code | 4x prose | 4x code | Paired vs base |
|---|---|---|---|---|---|---|---|
| prod (production binary) | 31.8 | 45.4 | 39.2 | 47.9 | 51.2 | 40.8 | -4% to +1% (4x code -10%, 1 of 3) |
| base | 31.5 | 44.5 | 40.0 | 43.5 | 50.8 | 45.3 | - |
| `STRATA_BATCH_DF2_CAP=1` | 31.9 | 45.0 | 41.4 | 47.0 | 53.4 | 47.8 | solo 0%; 4x +5% (3 of 3, both kinds) |
| `STRATA_DF2_Q4_NARROW=1` | 33.0 | 43.9 | 40.9 | 45.8 | 52.3 | 46.2 | -3% to +5%, mixed |
| `STRATA_PCIE_NEVER=1` | 32.0 | 44.6 | 40.1 | 46.2 | 52.3 | 46.6 | -1% to +3% |
| all three | 30.9 | 46.2 | 40.0 | 46.7 | 52.2 | 46.9 | solo -2%/+2%; batch +3% to +6% |
| `ded`: `--vram-reserve-later-mib 0` | 43.5 | 74.2 | 66.9 | 89.3 | 77.4 | 101.8 | solo +39%/+76%; batch +49% to +132% (every pair) |
| `ded25`: the same, `layer_split` 25 | 44.0 | 74.2 | 66.0 | 89.2 | 76.9 | 100.0 | the same as `ded` |

What it means:
- **The reserve for the second model is by far the biggest cost.** With 16.5 GB kept free, GPU 1 caches 4,396 of its 11,264 profiled experts (39%); without the reserve it caches them all. Solo code goes from 44.5 to 74.2 tok/s and 4 requests from 45 to 102 tok/s. Moving one more layer to GPU 1 (`layer_split` 25) changes nothing.
- The three new switches are small and within the noise of 3 runs: `STRATA_BATCH_DF2_CAP` gave +5% on 4 requests in every pair and nothing solo, as expected (it only trims the slot drafters' work). None of them made anything slower beyond noise.
- `prod` and `base` (the always-on host plan change) are level.
- The narrow Q4_0 drafter kernels matched the 8-column kernel bit for bit for 1, 2, 3 and 4 columns on GPU 1 (`dflash2_gemm_bench`).
- Greedy answers are not repeatable here even within one arm (0 of 4 repeats of the same essay identical): which experts the CPU or the GPU computes changes near-ties. Answer identity is therefore no check between arms on this rig; the kernel and plan checks above are.

### 9.6 Measured: round B, stopped early (2026-11-25)
The same setup as round A, with the watchdog fix in the build. Round B's `base` matched round A's within 0.5% (solo code 44.3 against 44.5, 4 requests of code 45.3 against 45.3). Paired against `base`:

| Arm | Solo prose | Solo code | 3x prose | 3x code | 4x prose | 4x code | Verdict |
|---|---|---|---|---|---|---|---|
| `STRATA_CPU_IQ_ALL_EXACT1=1` | +2.7% | +0.2% | +3.7% (3 of 3) | +7.6% | -0.3% | +5.3% (3 of 3) | small batch gain: kept |
| `STRATA_KQ_SPLIT=1` | +0.8% | -3.7% | -0.1% | +0.7% | +0.2% | -5.0% (0 of 3) | rejected |
| `STRATA_FUSE_HITS=1` | -1.4% | -11.2% (0 of 6) | -10.7% | -3.7% | -1.6% | -6.4% | rejected: slower |
| `STRATA_BATCH_SLOT_RUNS=1` | one run only: solo -3% to -11%, bursts -1% to -8% | | | | | | rejected (provisional) |

The round was stopped here to bring production back; `STRATA_PIPE_ADAPT_EVERY=0`, `STRATA_SH_STREAM=1` and round C (`STRATA_EXP_MODE=8`, `STRATA_BATCH_GDN_GROUPS`, `STRATA_POOL_DEP=0`, `--pool-tasks 12`/`24`) are not measured.

### 9.7 Production config from rounds A and B (2026-11-25)
- The new build (this branch: the watchdog fix, the plan map and the opt-in switches), installed beside the old binary, which stays for a rollback.
- Added to `env`: `STRATA_BATCH_DF2_CAP=1`, `STRATA_DF2_Q4_NARROW=1`, `STRATA_PCIE_NEVER=1` (measured together: batch +3% to +6%), `STRATA_CPU_IQ_ALL_EXACT1=1` (batch +4% to +5%), `STRATA_WATCHDOG_S=900` (section 10). The four speed switches were measured one round at a time, not all together.
- Unchanged: the second model and its 16.5 GB reserve on GPU 1. Dropping the reserve is the largest gain measured (round A, `ded`), but it needs the second model moved or stopped.
- Checked after the start on the production service: the engine log names `STRATA_BATCH_DF2_CAP` and `STRATA_PCIE_NEVER`; a 12,033-token prompt read at 799 tok/s; 4 code requests at once gave 22.6, 43.9, then 66.3 tok/s in all as the cache warmed; solo code 40, then 56 tok/s.

### 9.8 GPU 1 VRAM Reclamation (measured, 2026-11-25)
- **Root cause of the `16,500 MiB` reserve:** With free headroom left on GPU 1 for a co-resident model, Strata skipped prompt-buffer borrowing (`CUDA1 keeps its own prompt buffers: 1.82 GiB`) and silently allocated `~2.6 GiB` of FP16 prefill dense-weight cache (`gemm.cu` auto-caches whenever `>3 GiB` is free), dropping GPU 1 free VRAM by **`-2,730 MiB`** on the first prompt.
- **Fix (config only, zero changes to the co-resident model):** Set `STRATA_SPLIT_OWN_BUFFERS=0`, `STRATA_PF_W16_CACHE_MIB=0`, and lower `--vram-reserve-later-mib` from `16500` to `12000`.
- **Measured result:** GPU 1 post-prompt VRAM drop fell from **`-2,730 MiB` → `-174 MiB`**. GPU 1 expert cache grew from **`4,396` (39.0%, `8.82 GiB`) → `6,591 / 11,264` (58.5%, `13.21 GiB`, `+2,195` experts)** while leaving **`1,953 MiB` free** alongside the co-resident model (`~9.8–10.1 GiB` VRAM). On live 27k–35k-token requests, decode expert hit rate rose from `89–93%` to **`95.8–97.6%`** and live decode rose from `25–34 tok/s` to **`38–80 tok/s`**.
- **Rejected:** Shrinking the co-resident model's `--ubatch-size` (`16384 → 1024`) freed `~4 GiB` idle VRAM, but the model uses non-causal attention requiring `n_ubatch >= n_tokens` and aborts on real prompts up to `16k` tokens.

### 9.9 Phase 1: `gfx906` Wave64 Decode Kernel Rewrites (measured, 2026-11-25)
Opt-in `gfx906` wave64 rewrites of the four hot decode kernel families (`~86%` of batch GPU compute), compiled with `0 B` scratch spill across all hot paths. Default paths remain byte-identical when flags are unset.

| Opt-in Switch | Kernel Family & Mechanism | Parity / Accuracy | Measured MI50 Micro-Benchmark (`180 W`) |
|---|---|---|---|
| `STRATA_WAVE_MMVQ_V2=1` | Dense GEMVs & `Q6_K` LM head (`native_mmvq.cu`): unpack-once wave64 auto-dispatch (1-shot whole-X LDS with pre-converted `float d8` for small `X`, L1/L2 multi-row reuse for wide `X`, `__mul24`, `sched_barrier`) | **Bitwise exact** (`0` diffs across 18 shapes × 6 widths & `mmvq_multi_parity` 148,480 outputs) | `head Q6_K`: `T=1: 1.271→1.052 ms (-17.2%)`, `T=3: -16.1%`, `T=8: 5.492→4.701 ms (-14.4%)`; `gdn qkv/gate Q6_K`: `-11%` to `-46%`; `gdn out Q8_0`: `T=4: -40.7%`, `T=5: -37.9%`; `gdn out IQ4_XS`: `T=1: -25.8%` |
| `STRATA_EXP_MODE=11` *(or `=10`)* | Routed MoE experts (`iq_kernels.cu`): exact signed-byte `(g^m)+b` (`16→8` `sdot4`/32w), `__mul24`, 16-lane sub-warp fused gate/up+SwiGLU+Q8_1 with split even/odd tree (`0%` idle lanes at `n_embd=2560`, `55–65` VGPRs, `0 B` spill) + `LDS_DOWN_NT=4` down (`62–66` VGPRs) | **Bitwise exact** vs Mode 7 (`0` diffs across all 5 model quant pairs × `T=1,4,8` & `native_grouped_parity` 0 failures) | `iq3_s/iq4_nl`: `T=1: 201.4→178.2 µs (-11.5%)`, `T=4: 411.5→284.0 µs (-31.0%, gu -41.6%)`, `T=8: 793.6→558.4 µs (-29.6%, gu -40.8%)`; `iq3_xxs/iq4_nl`: `T=4/8: -28.3%/-26.3%`; `iq2_s/iq4_nl`: `T=4/8: -23.0%/-21.3%`; `iq4_xs/iq4_nl`: `T=4/8: -23.8%/-20.5%` |
| `STRATA_GR_SPLIT=3` *(`=2` exact FP32, `=4` hybrid)* | Hyper-connections (`fused_gr.cu`): 10-slice LDS down (`40 KiB` vs `64 KiB`) + wave64 FP16 `v_dot2_f32_f16` down & up (`61–63` VGPRs, `0 B` spill) | `=2`: exact FP32 (`rel RMS 4.5e-8`); `=3`: FP16 `mixed` rel RMS **`3.0–3.3e-5`**, `R` `0.00` exact | `COLD 8 sets` vs `=1`: `T=1: 47.1→37.8 µs (-19.7%)`, `T=2: -23.3%`, `T=4: 71.9→53.0 µs (-26.3%)`, `T=6: 96.3→63.7 µs (-33.9%)`, `T=8: 114.0→81.8 µs (-28.2%)` (`up` `-46.4%`) |
| `STRATA_MMVF_WAVE=1` + `STRATA_HIP_ROUTER_WAVE_FAST=1` | Router BF16 projection (`native_bf16.cu`) + wave64 top-10 (`router_top10.cu`); `STRATA_NATIVE_ROUTER_SPREAD=1` in `native_router.cu` measured neutral (`~11.9 µs`) | **Bitwise exact** (`0 / 65,536` top-10 diffs; `route_window_parity` 0 failures) | Top-10 (`512/k10`): `55.4→15.2 µs` (`3.63×` at `T=1..8`); BF16 `n_out=512`: `T=1: 45.0→9.9 µs (4.56×)`, `T=8 mmvf: 48.8→25.5 µs (-47.7%)`, `T=8 mmvf+route: 61.6→37.6 µs (-39.0%)` |

**Live End-to-End Dual-MI50 A/B (`layer_split=26`, `--vram-reserve-later-mib 12000`, `19,903` resident experts, warm cache `96–100%` hit, `256` tokens, `temp=0.0`, thinking off):**

| Workload | Baseline (`5f43512`) *(e2e / pure)* | **Phase 1 (5 flags)** *(e2e / pure)* | **Gain** |
|---|---:|---:|---:|
| **Solo Prose** | `43.3` / `47.5 tok/s` *(98.1% hit)* | **`49.7` / `54.3 tok/s`** *(98.5% hit)* | **`+14.8%`** e2e / **`+14.3%`** pure |
| **Solo Code** | `58.5` / `69.8 tok/s` *(96.1% hit)* | **`63.1` / `73.2 tok/s`** *(97.2% hit)* | **`+7.9%`** e2e / **`+4.9%`** pure *(cold warmup `+10.1%`)* |
| **3× Batch Prose** *(total)* | `63.0` / `71.3 tok/s` *(99.5–100% hit)* | **`72.2` / `81.4 tok/s`** *(100.0% hit)* | **`+14.6%`** e2e / **`+14.2%`** pure |
| **4× Batch Code** *(total / engine rows/s)* | `83.3` / `101.1` / `88.9 rows/s` *(92.7–97.3% hit)* | **`92.4` / `111.1` / `97.6 rows/s`** *(93.6–98.2% hit)* | **`+10.9%`** e2e / **`+9.9%`** pure / **`+9.8%`** rows/s |

### 9.10 Phase 2: Lossless VRAM Reclamation & Multi-GPU DFlash2 Tap Restore (measured, 2026-11-25)
Four opt-in lossless VRAM optimizations free `~2.5 GiB` on GPU 0 and `~2.1 GiB` on GPU 1 without reducing context length, batch concurrency, or `--vram-reserve-later-mib 12000`, enabling `layer_split` to move from `26 → 28` (`100%` zero-doorbell GPU 0 on layers `0–27`, `75.4%` GPU 1 residency on layers `28–47`, `90%` total model residency) while keeping `1,560 MiB` free on GPU 0 (elastic vision) and `12,276 MiB` free on GPU 1.

| Opt-in Switch / Fix | Mechanism | Verified VRAM Freed | Parity / Verification |
|---|---|---|---|
| `STRATA_SHARED_KV_CELLS=131072` | Per-stage chunk-paged `SharedKvPool` (256 pages = 1,024 cells/chunk) shared across solo + `--batch 4` slots via existing `page_table` indirection (zero kernel/graph changes); auto-reclaims idle cached slots and supports non-contiguous save/copy/restore (`conversation_snapshot.cpp`) | **`-2.31 GiB`** GPU 0, **`-1.68 GiB`** GPU 1 (`4` batch slots: `0.73→0.15 GiB` GPU 0, `0.53→0.11 GiB` GPU 1) | Non-contiguous `[0, 2, 4]` chunk save/copy/restore/verify exact hash match (`0` leaks); live `>6K`-token park (`115 ms`) and restore (`30.7 ms`) |
| `STRATA_LEAN_SLOTS=1` | Batch slot sessions share the main session's RoPE tables and omit unused single-token `block` / `gdn` / `moe` scratch | Included in `0.15 / 0.11 GiB` slot footprint (`~150 MiB/GPU`) | Default unchanged when unset |
| `STRATA_TRIM_VERIFY_SCRATCH=1` | Sizes verifier layer scratch (`qkv_L_`, `h_L_`, `dc_arena_`, snapshots) to stage-local layers only; omits `staging_` when `no_pcie`, `head_logits_` on non-final stages, `xil_` on HIP, and `hit_xq_`/`hit_xs_` on native layout | **`~140 MiB/GPU`** (verifier buffers `86.4 → 28.7 MiB` GPU 0, `32.5 MiB` GPU 1) | `layer_split=28` solo + `batch 4` verified |
| `STRATA_DF2_HEAD=indexed` | `native_mmvq_indexed` (`gfx906` wave64, `Q6_K`/`Q5_K`/`Q4_K`/`Q8_0`) indexes the 106,299 draft-vocab rows directly in the target head instead of duplicating them | **`-213 MiB`** GPU 1 (`625 → 412 MiB` drafter VRAM; `0.0 MiB` draft head) | **Bitwise exact** (`0` diffs vs gathered `native_mmvq` across `Q6_K/Q5_K/Q4_K/Q8_0` × `T=1,3,7`; `+22–197 µs`) |
| **Multi-GPU DFlash2 prompt taps + slot stash** *(always on)* | Fixed non-final prefill stage `continue;` skipping `on_taps(...)` (`prefill.cpp`); added non-P2P cross-GPU host bounce in `Df2PromptLand` (`generate.cpp`) to avoid ROCm 5.7 `hipMemcpy2DAsync` segfaults; stashed batch slot tap logs on park/evict/handoff (`dflash2.cpp`) | `0` VRAM | `dflash2_ring_test` pass; live `6,244`-token park/restore rebuilt **`2,022 / 2,022` rows (`0` missing, `0` cold starts) in `50.9 ms`** (was `46 / 2,022`), raising restored pure decode from `~43 → 64.2 tok/s` (`74.1%` draft acceptance) and cold 9.6K-prompt decode to `57.5 tok/s` (`73.0%` acceptance) |

**Live End-to-End Dual-MI50 Progression (`--vram-reserve-later-mib 12000`, `180 W`, warm cache, `256` tokens, `temp=0.0`, thinking off):**

| Workload | Phase 0 (`5f43512`, `split=26`, 81% res) | Phase 1 (`10c8655`, `split=26`, 81% res) | **Phase 2 (`split=28`, 90% res)** | Phase 1 → 2 | **Cumulative vs Phase 0** |
|---|---:|---:|---:|---:|---:|
| **Solo Prose** *(e2e / pure)* | `43.3` / `47.5 tok/s` | `49.7` / `54.3 tok/s` | **`52.3` / `57.1 tok/s`** | **`+5.2%`** / **`+5.2%`** | **`+20.8%`** / **`+20.2%`** |
| **Solo Code** *(e2e / pure)* | `58.5` / `69.8 tok/s` | `63.1` / `73.2 tok/s` | **`65.3` / `77.8 tok/s`** | **`+3.5%`** / **`+6.3%`** | **`+11.6%`** / **`+11.5%`** |
| **3× Batch Prose** *(total e2e)* | `63.0 tok/s` | `72.2 tok/s` | **`78.4 tok/s`** | **`+8.6%`** | **`+24.4%`** |
| **4× Batch Code** *(total e2e / engine rows/s)* | `83.3` / `88.9 rows/s` | `92.4` / `97.6 rows/s` | **`106.3` / `112.8 rows/s`** | **`+15.0%`** / **`+15.6%`** | **`+27.6%`** / **`+26.9%`** |
| **Resident Experts** *(total / GPU 1)* | `19,903 (81.0%)` / `58.5%` | `19,903 (81.0%)` / `58.5%` | **`22,054 (90.0%)` / `75.4%`** | **`+2,151`** experts | **`+16.9 pp`** GPU 1 |
| **Warm Decode Hit Rate** | `95.8–97.6%` | `96.0–99.5%` | **`99.0–99.6%`** | — | — |

### 9.11 Phase 3: SpecDecode & Pipeline Upgrades (measured, 2026-11-25)
Four opt-in spec-decode/pipeline upgrades in `generate.cpp`, paired with the verified HIP batch fusion flags (`STRATA_BATCH_SLOT_RUNS=1`, `STRATA_FUSE_HITS=1`). Default behavior is unchanged when flags are unset.

| Opt-in Switch | Mechanism | Parity / Verification |
|---|---|---|
| `STRATA_PW_PENALTY=1` *(+ neutral-penalty skip)* | Enables solo `pl_pw >= 2` for requests with `penalty_last_n > 0` by staging `penalty_rows` to `d_hist` before strictly in-order Stage 1 `V1(A).pl_launch`; neutral penalties (`repeat=1, freq=0, present=0`) skip `d_hist` entirely | Exact token-for-token output match vs serial `pw=0` (`frequency_penalty=0.25, presence_penalty=0.2`); `+12–18%` on penalized solo decode |
| `STRATA_PW_FORK_B=1` | Bonus-fork speculative window `B` in solo `pw=2`: when re-drafting (`df2_kind == 1`) disagrees with `A.tok[m+1]` at first mismatch `m`, speculatively commits `A`'s first `m+1` rows on Stage 0 and launches `B` starting at `dd[m]` while `A` verifies on Stage 1 (turning forced-chain disagreements into on-path hits; exact snapshot rollback on miss) | Exact greedy output match; forced-chain disagreements dropped to `0–1`, solo code speculative on-path rate rose to `57%` (`26.1 ms` / `3.08 tok` = `118 tok/s` per speculative window) |
| `STRATA_DF2_PW_DYN=1` | Dynamic `A`/`B` split sizing for fresh DFlash2 blocks (`df2_kind == 2`): chooses split point `t_a ∈ [2..5]` (or unsplit) maximizing expected committed tokens per stage-cost unit from `pp[0..6]` instead of fixed `half=4` | Exact greedy output match |
| `STRATA_BATCH_DYN_GROUPS=1` | Dynamic carrier balancing for `STRATA_PIPED_MTP`: keeps home-affine static groups at `3×`/`4×` (zero extra HIP graph captures), and splits active slots `1+1` across carriers whenever a static group would otherwise go empty (e.g., `2×` batch or staggered-finish tails) | Eliminates `50%` 2-GPU pipeline bubble on same-group pairs (`batch2_code`: `89.2 → 96.2 tok/s`) |
| `STRATA_BATCH_SLOT_RUNS=1` + `STRATA_FUSE_HITS=1` | Existing HIP fusion switches verified on `gfx906`: fuses contiguous per-slot QSA rows in `batch_launch_rows` and folds `moe_hit_add` into MoE combine (`-48` launches/window) | Zero NaNs, identical openings; `+4–10%` additional batch throughput |

**Live End-to-End Dual-MI50 Progression (`--vram-reserve-later-mib 12000`, `180 W`, warm cache, `256` tokens, `temp=0.0`, thinking off):**

| Workload | Phase 0 (`5f43512`) | Phase 1 (`10c8655`) | Phase 2 (`1450115`) | **Phase 3 (fresh / warm)** | **Cumulative vs Phase 0** |
|---|---:|---:|---:|---:|---:|
| **Solo Prose** *(e2e / pure)* | `43.3` / `47.5 tok/s` | `49.7` / `54.3 tok/s` | `52.3` / `57.1 tok/s` | **`56.9` / `62.6 tok/s`** | **`+31.4%`** / **`+31.8%`** |
| **Solo Code** *(e2e / pure)* | `58.5` / `69.8 tok/s` | `63.1` / `73.2 tok/s` | `65.3` / `77.8 tok/s` | **`73.7` / `87.5 tok/s`** | **`+26.0%`** / **`+25.4%`** |
| **Solo Penalized Prose** *(e2e / pure)* | — | — | `46.2` / `51.5 tok/s` | **`51.8–54.4` / `55.6–58.7 tok/s`** | **`+12–18%`** vs Phase 2 |
| **2× Batch Code** *(total / engine rows/s)* | — | — | `89.2` / `98.1 rows/s` | **`96.2` / `106.0 rows/s`** | **`+7.8%`** vs Phase 2 |
| **3× Batch Prose** *(total / engine rows/s)* | `63.0 tok/s` | `72.2 tok/s` | `78.4` / `82.6 rows/s` | **`84.6` / `89.4 rows/s`** | **`+34.3%`** |
| **4× Batch Code** *(total / engine rows/s)* | `83.3` / `88.9 rows/s` | `92.4` / `97.6 rows/s` | `106.3` / `112.8 rows/s` | **`110.9–120.0` / `118.6–129.1 rows/s`** | **`+33.1–44.1%`** e2e / **`+33.4–45.2%`** rows/s |

### 9.12 Phase 4: Prefill Rework & `gfx906` rocBLAS Tuning (measured, 2026-11-25)
Five opt-in `gfx906` prefill upgrades targeting the four dominant MI50 prompt bottlenecks (untuned FP16 dense GEMMs, `<32 KiB` LDS occupancy in routed `native_expert_i8`, per-layer host sync + 16-expert group splits, and FP32/LDS-bound `k8v4` QSA prompt attention), plus a ROCm 5.7 `Optin` shared-memory attribute fallback fix (`strata_hip.h` / `i8_expert.cuh`). Default behavior is unchanged when flags are unset.

| Opt-in Switch / Fix | Mechanism | Parity / Verification | Measured MI50 Micro-Benchmark (`180 W`) |
|---|---|---|---|
| `STRATA_ROCBLAS_TUNING=<table.txt>` *(+ CMake `STRATA_ROCBLAS_AVAILABLE` for `gfx906`)* | Enables `STRATA_ROCBLAS_AVAILABLE` and `gfx906_prefill_rocblas_gemm` in `backends/hip.cmake` / `strata_gpu_engine.cmake`; includes tuned `gfx906` `rocBLAS 2.47.0` table (`terminal/tools/hip/gfx906-rocblas-2.47.0.txt`, `98` rows / `88` solutions) for `STRATA_HIP_PROMPT_F16=1` | `gfx906_prefill_rocblas_gemm` `PASS` (`4/4` cases, `rel_l2=2.08e-4`, `tuned_launches=4, fallbacks=0`) | **`1.43×–3.35×`** vs default `f16/bf16->f32` (`1.67×–4.06×` vs untuned `f16o`): `hc_down` `4.55→1.36 ms (3.35×)`, `hc_up` `2.37→0.82 ms (2.91×)`, `router` `1.08→0.46 ms (2.32×)`, `attn_q` `11.07→6.36 ms (1.74×)`, `gdn_qkv` `8.83→5.35 ms (1.65×)` |
| `STRATA_I8_TM=7` *(+ ROCm 5.7 64 KiB `i8_smem_limit()` fix)* | New `<32 KiB` single-buffered LDS wave64 `i8w_kernel` (`i8_expert.cuh`: `128×64` tile, `20.2–24.3 KiB` LDS, 2 blocks/CU, `127–128` VGPRs, **`0` spills / `0 B` private scratch** across all model quant types); fixes ROCm 5.7 `hipDeviceAttributeSharedMemPerBlockOptin` error returning `48 KiB` instead of `64 KiB` | **Bitwise exact** output hashes vs baseline `TM=2/3/4/8`; `native_i8_parity` `all ok` (`70+` quant pairs) | `IQ3_S/IQ4_NL`: `T=2048: 22.54→18.81 ms (1.20×)`, `T=4096: 46.19→36.83 ms (1.25×)`; `IQ2_S/IQ4_NL`: `T=2048: 29.99→19.36 ms (1.55× vs pre-fix TM=2)`, `T=4096: 46.56→37.16 ms (1.25×)`; `IQ3_XXS/IQ4_NL`: `T=2048: 22.90→18.42 ms (1.24×)` |
| `STRATA_PF_I8_GPU_PLAN=1` | On-GPU routing plan via `fused::group()` inside `m.GU` tail (`0` extra VRAM, `prefill.cpp` + `moe_fused.hpp`): eliminates per-layer `cudaStreamSynchronize`, D2H/H2D routing copies, and `32×` 16-expert group splits (launches all `512` resident experts in `1` call on GPU 0 and `ring/3`-streamed batches on GPU 1) | Verified live activation + coherent outputs on `pp4k`–`pp29k` (`0` errors) | Eliminates `48` host syncs + `~1,500` small group launches per 2K chunk |
| `STRATA_QSA_PA_GFX906=1` | Single-pass online-softmax wave64 `prompt_attn_gfx906_kernel` (`qsa_prompt_attn.cu`, `k8v4` & `int8`): power-of-2 prescaled FP16 Q (`6 KiB` LDS) + cell-paired FP16 P×V via hardware `v_dot2_f32_f16` (`11.4 KiB` LDS, `0` spills, `0` split-K scratch/merge launches) | `qsa_parity` ok; standalone `n_q=2048, cap=2051`: `nan=0`, `k8v4 rel_l2=3.08e-4`, `int8 rel_l2=3.90e-4` | `n_q=2048, cap=2051` shared block-4 IDs: `k8v4` `58.55→46.95 ms` (**`1.25×`**), `int8` `58.95→45.25 ms` (**`1.30×`**) |
| `STRATA_ATTN_MERGE_V2=1` + existing GDN flags | Batched attention merge (bitwise exact); verified existing `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_NOY=1` on `gfx906_gdn_rec_head` (`7.80→3.24 ms/layer`, `2.41×`, `0` diffs). Note: `gdn_chunk.cu` targets Volta WMMA (`>64 KiB` LDS) and no container runtime was available on the host for ROCm 6.3+/7.x A/B | Bitwise exact (`FAILURES: 0`) | `~164 ms`/chunk saved by GDN recurrence flags (already enabled in profile) |

**Live End-to-End Dual-MI50 Prefill & Decode Summary (Phases 0 → 4, `--vram-reserve-later-mib 12000`, `180 W`, `temp=0.0`, thinking off):**

| Workload | Phase 0 (`5f43512`) | Phase 3 (`7416972`) | **Phase 4 (live)** | **Phase 3 → 4** | **Cumulative vs Phase 0** |
|---|---:|---:|---:|---:|---:|
| **Prefill `pp4k` (`7,328` tok, 0% reuse)** *(pure / e2e)* | — | `453.8` / `451.1 tok/s` (`16.15 s`) | **`882.5` / `732.8 tok/s`** (`8.30 s`) | **`+94.5%`** (`1.94×` pure) | — |
| **Prefill `pp9k` (`9,318` tok, 0% reuse, warm)** *(pure / e2e)* | — | — | **`950.2` / `939.9 tok/s`** (`9.81 s`) | — | — |
| **Prefill `pp12k` (`22,588` tok, 0% reuse)** *(pure / e2e)* | — | `763.4` / `760.8 tok/s` (`29.59 s`) | **`995.5` / `990.4 tok/s`** (`22.69 s`) | **`+30.4%`** (`1.30×` pure) | — |
| **Prefill `pp29k` (`28,818` tok, 0% reuse, warm)** *(pure / e2e)* | — | — | **`937.3` / `933.9 tok/s`** (`30.74 s`) | — | — |
| **Solo Prose** *(e2e)* | `43.3 tok/s` | `56.9 tok/s` | **`58.6 tok/s`** | `+3.0%` | **`+35.3%`** |
| **Solo Code** *(e2e / pure)* | `58.5` / `69.8 tok/s` | `73.7` / `87.5 tok/s` | **`71.1–73.6` / `90.3 tok/s`** | level / `+3.2%` pure | **`+25.8%`** / **`+29.4%`** |
| **Solo Penalized Prose** *(e2e / pure)* | — | `51.8–54.4` / `55.6–58.7 tok/s` | **`54.7` / `59.0 tok/s`** | `+0.6%` | **`+18.4%`** vs Phase 2 |
| **2× Batch Code** *(total / engine rows/s)* | — | `96.2` / `106.0 rows/s` | **`98.7` / `109.3 rows/s`** | `+2.6%` | **`+10.7%`** vs Phase 2 |
| **3× Batch Prose** *(total e2e)* | `63.0 tok/s` | `84.6 tok/s` | **`82.7 tok/s`** | within run noise | **`+31.3%`** |
| **4× Batch Code** *(total / engine rows/s)* | `83.3` / `88.9 rows/s` | `110.9–120.0` / `118.6–129.1 rows/s` | **`117.3` / `120.1 rows/s`** | level | **`+40.8%`** / **`+35.1%`** |
| **Post-Prefill Free VRAM** *(GPU 0 / GPU 1)* | `~1.7` / `~1.9 GiB` | `1.56` / `12.0 GiB` | **`1,276 MiB` / `11,929 MiB`** | `12 GiB` reserve intact | — |

## 10. Production Incident: Nightly Backup Stalls the VM's Disk (measured, 2026-11-25)
The rig is a KVM guest. Its hypervisor backs it up every night in snapshot mode. The backup freezes the guest file system for a moment (the QEMU guest agent logs `guest-fsfreeze`, around 23:00-23:15), then copies the disk while the VM runs. In that mode a guest write to a block that is not yet backed up waits until the old block has reached the backup target ("copy-before-write"). When the backup target is slow, every write in the guest waits.

What the guest's logs show for the night of 2026-11-25:
- From about 00:46 no write finished for about 10 minutes. `systemd-journald` and `systemd-timesyncd` hit their 3-minute watchdogs; journald could not be killed until 00:56:49. The same journald watchdog kill is in the journal on Nov 22 (00:40) and Nov 23 (00:37): it happens most nights.
- The admission service's records of the five requests that ended during the stall all carry 00:56:49, so its SQLite writes waited too. (The night's earlier deadline failures, from 23:45 on, fall inside the backup window as well; whether the backup slowed them is not shown.)
- Strata kept serving through the stall: two short and two long requests finished, and a 9,269-token answer ran at 31 tok/s until the client's 300 s deadline cancelled it. The next request then stood still for about 3 minutes while it parked the previous conversation (most likely on a write: the GPU and the CPU expert pool were idle). The engine's watchdog (issue #29, `STRATA_WATCHDOG_S`, default 60 s) saw no heartbeat, but its own message to the engine log on the same disk was stuck too. It got out when the writes resumed and stopped the engine 7 s later, although the stall report showed every GPU stream and every CPU expert job finished. The park itself then took 111 ms.
- The client stopped sending after that error; no request reached the admission service for the next 4 hours.

Fixes:
1. **On the hypervisor: backup fleecing.** The old blocks go to a local fleecing image instead of the backup target, so guest writes no longer wait on the backup (Proxmox: the backup job's `fleecing: enabled=1,storage=<fast local storage>`). Keeping the model files out of the backup shortens the window too. This is the fix that also covers the journal and the admission service.
2. **Engine watchdog (this branch, `src/program/generate.cpp`):** a system-wide storage stall (Linux PSI, `/proc/pressure/io`, the "full" total grows by 0.2 s or more in a second) now counts like the file tier's reads (#1407): slow, not stuck, within `STRATA_WATCHDOG_IO_S` (default 10 x the limit). After the stall report, which takes 2 s or more, the watchdog looks again; a request that moved meanwhile is kept, and nothing is released or stopped. Behaviour changes only while the system is stalled on storage or the request moves during the report.
3. **Until that build is in production:** `STRATA_WATCHDOG_S=900` in the production profile's `env`. A real GPU hang then takes up to 15 minutes to recover instead of 1.
4. **The client** should retry after a backend error instead of stopping for the night.
