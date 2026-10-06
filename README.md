<div align="center">

# Jordan Anderson

### Faster MoE training · larger models · constrained-GPU execution

**Qwen3-30B-A3B on an RTX 5090: 2.35× training speed vs Unsloth at matched work.**

3.494 vs 8.218 s/step on the same torch/transformers stack, with comparable held-out loss.  
Replicated on a second host at **2.468×**.

<p>
  <a href="https://github.com/pjordanandrsn/loggetta"><img src="https://img.shields.io/badge/Loggetta-full%20stack-6F42C1" alt="Loggetta"></a>
  <a href="https://pypi.org/project/loggetta/"><img src="https://img.shields.io/pypi/v/loggetta?label=loggetta" alt="loggetta"></a>
  <a href="https://pypi.org/project/experts4bit-qlora/"><img src="https://img.shields.io/pypi/v/experts4bit-qlora?label=runtime" alt="experts4bit-qlora"></a>
  <a href="https://pypi.org/project/grouped-nf4-gemm/"><img src="https://img.shields.io/pypi/v/grouped-nf4-gemm?label=kernels" alt="grouped-nf4-gemm"></a>
</p>

<p>
  <a href="https://cerinamroth.com">Research notes</a> ·
  <a href="https://jordananderson.work">Work</a> ·
  <a href="https://cerinamroth.com/policy/">Security policy</a>
</p>

</div>

---

I build the missing layer between **“this should fit”** and **“here is what actually ran.”**

My current work is a small systems stack for running and fine-tuning large Mixture-of-Experts models on hardware far smaller than the model.

## Measured speed

| Result | Comparison |
| :--- | :--- |
| **2.352× vs Unsloth** | Qwen3-30B-A3B QLoRA on one RTX 5090, matched adapters/init/tokens and the same torch 2.12.1+cu130 / transformers 5.5.0 stack: **3.494 vs 8.218 s/step**. Held-out loss at N=60 was **0.7569 vs 0.7557** and graded COMPARABLE. Unsloth used less peak VRAM: **24.27 vs 27.49 GB**. |
| **2.468× replication** | Same-stack Qwen3-30B-A3B comparison on a second RTX 5090 host: e4b **4.143 / 4.073 s/step** vs Unsloth **10.155 / 10.120 s/step**. |
| **2.775× vs Axolotl** | Matched-work Qwen3-30B-A3B comparison on an RTX 5090 / Ryzen 9 9950X3D host: **2.147 vs 5.956 s/step**. A separate EPYC-host reading was **1.979×**, so the ratio is explicitly host-sensitive. |
| **2.33× packed-compute throughput** | H100 synthetic expert-offload pipeline vs bitsandbytes CUDA dequantization + cuBLAS: **6.466 vs 2.773 pipeline tok/s**, **26.8 vs 59.1 J/token**. This is a pipeline result, not an end-to-end serving claim. |

Evidence: [same-stack Unsloth result](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-samestack-box4.md) · [second-host replication](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/changelog.d/tc1-amendment-42-read-samestack-host2.md) · [matched Axolotl/Unsloth result](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-matched19.md) · [H100 bnb comparison](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase3/flagship/RESULTS-flagship-bnb-baseline.md)

> These are scoped measurements of the runtime/kernel stack behind Loggetta, not speedups attributed to the planner. Ratios are within-box readings, not numbers divided across machines.

## Start with Loggetta

**[Loggetta](https://github.com/pjordanandrsn/loggetta) is the user-facing home for the stack.**

```bash
pip install loggetta
loggetta inspect
loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
```

That one install brings in:

- **Loggetta** — planning, `ExecutionPlan`, execution orchestration, and `ExecutionReceipt`
- **experts4bit-qlora** — model loading, QLoRA, training, serving, and residency/offload
- **grouped-nf4-gemm** — packed low-bit kernels and low-level residency primitives

Loggetta examines the model, workload, machine, and constraints **before loading model weights**. It selects a supported configuration, explains why alternatives lost, and refuses configurations estimated not to fit.

A feasible plan is still an estimate, not an out-of-memory guarantee.

### The loop

```mermaid
flowchart TB
    U["User: pip install loggetta"] --> P
    I["Model + workload + machine<br/>constraints + evidence"] --> P
    subgraph L["Loggetta"]
        P["Planner"] --> X["ExecutionPlan<br/>what should run + why"]
        X --> O["validate + dispatch"]
        R["ExecutionReceipt<br/>what actually happened"] -. "measured feedback" .-> P
    end
    O --> E["experts4bit-qlora<br/>runtime"]
    E --> G["grouped-nf4-gemm<br/>kernels + residency"]
    E -. "execution results" .-> R
```

An **ExecutionReceipt** is the run report: estimated vs measured memory, timing, correctness checks, checks that the selected optimizations actually ran, and the code/runtime provenance behind the result.

> **Loggetta decides what should execute. The runtime knows how to execute it.**

## Current Loggetta scope

Released today:

- single-GPU MoE planning
- saved, inspectable execution plans
- QLoRA training execution through the included runtime
- measured run reports that can inform later plans
- serving placement planning across device, host, and storage tiers
- one default install for the planner + runtime + kernel stack

Not claimed yet:

- first-class dense-model planning
- multi-GPU placement/execution planning
- calibrated throughput prediction
- Loggetta server launch
- universal exposure of every lower-layer capability

### In development: your data → reusable adapters

**[Loggetta 0.2.0 candidate, PR #2](https://github.com/pjordanandrsn/loggetta/pull/2)** is implemented and both CI lanes are green, but it is not yet the released PyPI build.

The new path adds:

- local JSON / JSONL / CSV / Parquet / TXT or Hugging Face datasets
- text, instruction/response, and chat formats
- deterministic shuffle and explicit data repetition
- dataset and learning-rate choices preserved in saved plans
- data validation and tokenization before model loading
- reusable adapter-only safetensors
- tokenizer files, manifest, runtime setup, base revision when known, and checksums
- strict reload validation and no silent overwrite

Local validation: **81 passed, 6 skipped**. CPU LoRA train → save → reload reproduces the trained logits while frozen parameters remain unchanged. Mixed fused-expert + attention adapters also round-trip correctly.

> The remaining gate is the full CUDA train/save/reload integration test. The adapters are a native runtime format, not a claimed PEFT interchange format.

## Selected measured results

| Experiment | Result |
| :--- | :--- |
| **gpt-oss-120b QLoRA on native MXFP4** | **9.82 GB peak VRAM**, with **144/144** native expert tensors hash-identical through training. |
| **30B-class expert offload** | **Qwen3-30B-A3B: 7.16 GB** training peak; **Gemma-4-26B-A4B: 8.47 GB** in the recorded offload runs. |
| **Loggetta, RTX 5090 / Qwen3-30B-A3B** | Allocator estimate **22.09 GiB**, measured **21.91 GiB**. After receipt calibration: planned process peak **24.54 GiB**, measured **24.34 GiB**. |
| **Loggetta serving placement, A2000 / OLMoE** | Planned VRAM / RAM / NVMe split matched the server: **272 / 421 / 331 experts**; allocator **2.052 GiB planned vs 2.051 GiB measured**. |
| **Packed compute vs bitsandbytes CUDA dequant** | **2.33× throughput**, **26.8 vs 59.1 J/token** on the same H100 synthetic offload pipeline. |

These are scoped experiments, not a single benchmark or universal speed/memory claim.

Evidence:
[Loggetta results](https://github.com/pjordanandrsn/loggetta/blob/main/docs/RESULTS.md) ·
[e4b provenance](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/PROVENANCE.md) ·
[120B MXFP4 training result](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/mxfp4/RESULTS-mxfp4-train.md) ·
[H100 bnb comparison](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase3/flagship/RESULTS-flagship-bnb-baseline.md)

## Receipts-driven engineering

I treat measurement infrastructure as part of the system.

| Principle | Practice |
| :--- | :--- |
| **Scope every claim** | Hardware, driver, software stack, workload, and exact code revision travel with the result. |
| **Register important comparisons first** | Protocols are OpenTimestamps-stamped before the outcome is known. |
| **Publish refutations** | Failed confirmatory runs stay public. |
| **Check invariants** | Frozen weights are hash-checked; selected optimizations are checked for actual engagement. |
| **Label uncertainty** | Measured, derived, inferred, and heuristic quantities remain distinct. No invented throughput numbers. |

[**Receipts-driven engineering**](https://cerinamroth.com/ml/grouped-nf4-gemm/)

## Upstream work

### MoE ecosystem

- [bitsandbytes #1965](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965) — `Experts4bit` for fused 4-bit MoE experts
- [Axolotl #3797](https://github.com/axolotl-ai-cloud/axolotl/pull/3797) — `expert_offload` integration
- [Unsloth Zoo #915](https://github.com/unslothai/unsloth-zoo/pull/915) — OLMoE fused-expert routing fix
- [Unsloth Zoo #849](https://github.com/unslothai/unsloth-zoo/issues/849) — silent expert-weight transposition, reproduced and fixed upstream

### Intel GPU / OpenVINO

Merged kernel-compile fixes across LoRA, MoE, and fully-connected paths:
[#35661](https://github.com/openvinotoolkit/openvino/pull/35661),
[#35712](https://github.com/openvinotoolkit/openvino/pull/35712), and
[#36017](https://github.com/openvinotoolkit/openvino/pull/36017).

[#36543](https://github.com/openvinotoolkit/openvino/pull/36543) extends the input-validation work.

[ov-impact-bench](https://github.com/pjordanandrsn/ov-impact-bench) measures GPU-vs-CPU-fallback impact on Intel hardware.

## Elsewhere

Security research, good-faith testing, and coordinated disclosure: [policy](https://cerinamroth.com/policy/).

<div align="center">

[**cerinamroth.com**](https://cerinamroth.com) · [**jordananderson.work**](https://jordananderson.work)

**Plan. Run. Measure. Keep the evidence.**

</div>