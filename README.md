<div align="center">

# Jordan Anderson

**ML systems · GPU kernels · evidence-led research**

![Large models. Smaller machines. Plan, run, measure, and keep the evidence.](assets/research-banner.svg)

[Research](https://cerinamroth.com/ml/) · [Technical & policy work](https://jordananderson.work) · [Hugging Face](https://huggingface.co/pjordanandrsn)

<p><a href="https://pypi.org/project/loggetta/"><img src="https://img.shields.io/pypi/v/loggetta?label=Loggetta&amp;color=6f8f9f&amp;style=flat-square" alt="Loggetta on PyPI"></a> <a href="https://pypi.org/project/experts4bit-qlora/"><img src="https://img.shields.io/pypi/v/experts4bit-qlora?label=runtime&amp;color=6f8f9f&amp;style=flat-square" alt="Runtime on PyPI"></a> <a href="https://pypi.org/project/grouped-nf4-gemm/"><img src="https://img.shields.io/pypi/v/grouped-nf4-gemm?label=kernels&amp;color=6f8f9f&amp;style=flat-square" alt="Kernels on PyPI"></a></p>

</div>

I build systems that make large Mixture-of-Experts models useful on the hardware people have: low-bit training, packed GPU compute, and execution across VRAM, host memory, and storage. The plan, the actual run, and the evidence belong together.

## Start with Loggetta

**One install for the planner, runtime, and kernels.**

```bash
pip install loggetta
loggetta inspect
loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
```

[Loggetta](https://github.com/pjordanandrsn/loggetta) examines the machine and workload before model weights load. It chooses among supported configurations, explains rejected alternatives, executes supported QLoRA training, and keeps an `ExecutionReceipt` alongside the `ExecutionPlan`.

| Layer | What I build |
| :--- | :--- |
| **[Loggetta](https://github.com/pjordanandrsn/loggetta)** | Hardware-aware planning, explicit refusals, execution orchestration, and measured feedback. |
| **[experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora)** | Streaming model loading, fused-expert QLoRA, training, paged serving, and CPU/NVMe residency. |
| **[grouped-nf4-gemm](https://github.com/pjordanandrsn/grouped-nf4-gemm)** | Triton kernels for packed NF4/MXFP4 compute, INT4 decode, FP8 paged attention, and residency primitives. |

**Released:** Loggetta **0.1.3**, e4b **0.48.0**, gnf4 **0.41.0**. Loggetta currently plans single-GPU MoE workloads and executes supported training; serving launch remains a runtime step. Memory estimates have measured errors, and calibrated throughput prediction is still open.

## Measurements worth inspecting

| Experiment | Result and tradeoff |
| :--- | :--- |
| **Qwen3-30B-A3B · RTX 5090 · same software stack** | **2.352×** training speed vs Unsloth: **3.494 vs 8.218 s/step**, matched adapters, initialization, and tokens; comparable held-out loss. Unsloth used **3.22 GB less peak VRAM**. [Result](https://github.com/pjordanandrsn/experts4bit-qlora/blob/87fe75f13f5b36d96aa2b109a66b76a55080efd7/bench/h2h-2026-10-02/tc1/RESULTS-tc1-samestack-box4.md) |
| **Second RTX 5090 host** | **2.468×** on the same comparison recipe. A second-host replication is quoted separately from the registered position. [Replication](https://github.com/pjordanandrsn/experts4bit-qlora/blob/87fe75f13f5b36d96aa2b109a66b76a55080efd7/bench/h2h-2026-10-02/tc1/RESULTS-tc1-samestack-host2.md) |
| **120B native-MXFP4 QLoRA experiment** | **9.82 GB peak VRAM**, **144/144** frozen expert tensors hash-identical through training. This lower-layer experiment is separate from Loggetta's supported training surface. [Evidence](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/mxfp4/RESULTS-mxfp4-train.md) |

The speed comparisons are within-box measurements of the runtime and kernels, with both frameworks on torch 2.12.1+cu130 / transformers 5.5.0. Hardware, workload, loss, and memory tradeoffs travel with each result.

<details>
<summary><strong>Latest research · 6 October 2026</strong></summary>

- **A different model gives a different ratio.** Same-stack Mixtral-8x7B training measured **1.144×** vs Unsloth, with comparable held-out loss. Unsloth used **2.07 GB less peak VRAM** and less energy per step. [TC2 result](https://github.com/pjordanandrsn/experts4bit-qlora/blob/87fe75f13f5b36d96aa2b109a66b76a55080efd7/bench/h2h-2026-10-02/tc2/RESULTS-tc2-mixtral-samestack.md)
- **Packed 4K context needs its own recipe.** Qwen3's opt-in `E4B_CHUNKED_LM_LOSS=1` run measured **1.278×** vs Unsloth with nearly equal held-out loss. e4b used more peak VRAM; its defaults still OOM on that recorded packed workload. This is research after the 0.48.0 release. [TC1 result](https://github.com/pjordanandrsn/experts4bit-qlora/blob/87fe75f13f5b36d96aa2b109a66b76a55080efd7/bench/h2h-2026-10-02/tc1/RESULTS-tc1-packed4k-chunked2.md)
- **A kernel win can disappear in the training step.** The decoded grouped route failed the registered OLMoE step-speed gate and remains opt-in. [End-to-end reading](https://github.com/pjordanandrsn/experts4bit-qlora/blob/87fe75f13f5b36d96aa2b109a66b76a55080efd7/bench/h2h-2026-10-02/tc1/RESULTS-tc1-decoded.md)
- **Per-card rules matter.** The row-aware split-K term failed its registered keep rule on the L4 and A4000. Those results retain the losses alongside the wins. [L4](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/d3e7788939fbad98a63a8c06badd253764c6dac0/kernel/RESULTS-k30-splitk-r-term-l4.md) · [A4000](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/d3e7788939fbad98a63a8c06badd253764c6dac0/kernel/RESULTS-k32-splitk-r-term-a4000.md)
- **Your data → reusable adapters is a candidate.** [Loggetta 0.2.0, PR #2](https://github.com/pjordanandrsn/loggetta/pull/2) adds local/Hub data, pre-load validation, and native adapter-only safetensors with manifests and checksums. Both CI lanes pass; the full CUDA train/save/reload gate remains unverified. It is not the published 0.1.3 build or a claimed PEFT interchange format.

[Read the dated research digest](https://cerinamroth.com/research/#update-2026-10-06) · [Loggetta's plan-versus-run evidence](https://github.com/pjordanandrsn/loggetta/blob/dd4783fd86942bc3bd82a1fc2008a10d8757d7b6/docs/RESULTS.md)

</details>

## How I work

**Scope the claim. Register the important comparison. Run the control. Publish the failure.**

Receipts preserve code revisions, hardware, memory, timings, and checks that the selected optimization actually ran. Frozen packed weights are hash-checked. Measured, derived, inferred, and heuristic quantities stay distinct. Failed gates remain visible.

[Receipts-driven engineering](https://cerinamroth.com/research/receipts-driven-engineering/) · [Runtime claims register](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/claims.json) · [Kernel claims register](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/claims.json)

## Contact securely

[Download my public GPG key](https://jordananderson.work/.well-known/jordananderson-pubkey.asc) · [Verify on my website](https://jordananderson.work/#public-key) · [Repository copy](assets/jordananderson-pubkey.asc)

Ed25519 · expires **11 June 2028**. Primary fingerprint:

```text
DAEF C03B 9188 793C 80E5 20C8 68F0 6663 C7FB 250F
```

## Beyond the stack

- **Upstream systems work:** [OpenVINO GPU kernel fixes](https://cerinamroth.com/ml/openvino/), plus merged fixes in [bitsandbytes #1999](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1999), [Transformers #47931](https://github.com/huggingface/transformers/pull/47931), and [Unsloth Zoo #1082](https://github.com/unslothai/unsloth-zoo/pull/1082).
- **Astronomy:** an [ongoing search of public JWST images](https://cerinamroth.com/jwst/lrds/), with observations and evidence attached to individual sources.
- **Software security:** [good-faith research and coordinated disclosure](https://cerinamroth.com/policy/).
- **Technical and policy consulting:** [selected work and contact](https://jordananderson.work), including industrial capacity and institutional design.

<div align="center">

**Plan. Run. Measure. Keep the evidence.**

[cerinamroth.com](https://cerinamroth.com) · [jordananderson.work](https://jordananderson.work)

</div>
