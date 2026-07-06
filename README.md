# Jordan Anderson

I find the gap between what an institution's contract claims and what its implementation actually does, and write the patch.

Current arc: **QLoRA fine-tuning of large Mixture-of-Experts models on small GPUs** — 4-bit-quantize the fused expert stack, stream it from pinned CPU RAM, one layer resident:

- [`experts4bit-qlora`](https://github.com/pjordanandrsn/experts4bit-qlora) — the working stack, [on PyPI](https://pypi.org/project/experts4bit-qlora/) (v0.2.0). Fills the hole where bitsandbytes' 4-bit walker silently skips transformers v5's fused experts ([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)). Qwen3-30B-A3B QLoRA peaks at 7.16 GB on a 12 GB card, Gemma-4-26B-A4B at 8.47 GB — both OOM without it. Matched A/B telemetry (shared seed + data order, per-step wandb/JSONL) in [`ab-telemetry/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/main/ab-telemetry): −57 % peak VRAM, convergence preserved.
- [bitsandbytes#1965](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965) (open) — `Experts4bit` upstreamed: NF4/FP4 quantization of the fused MoE expert parameter.
- [axolotl#3797](https://github.com/axolotl-ai-cloud/axolotl/pull/3797) (open) — `expert_offload`: the offload path as a self-contained axolotl integration (single-GPU, gated behind `expert_offload: true`).

Before that: OpenVINO Intel GPU plugin — `__local`-pointer kernel-compile fixes across the LoRA / MoE / fully-connected kernels, plus a regression test so the bug class can't silently return ([#35661](https://github.com/openvinotoolkit/openvino/pull/35661), [#35712](https://github.com/openvinotoolkit/openvino/pull/35712), [#36017](https://github.com/openvinotoolkit/openvino/pull/36017) — all merged). [ov-impact-bench](https://github.com/pjordanandrsn/ov-impact-bench) measures what the fixes unlock — latency, energy, throughput — on real Intel silicon.

Security research on the side.

More at [jordananderson.work](https://jordananderson.work).
