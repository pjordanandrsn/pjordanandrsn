<div align="center">

# Jordan Anderson

**I make large models run on smaller machines.**

![Large models. Smaller machines. Plan, run, measure, and keep the evidence.](assets/research-banner.svg)

[Research](https://cerinamroth.com/ml/) · [Work with me](https://jordananderson.work) · [Hugging Face](https://huggingface.co/pjordanandrsn)

</div>

I build open-source ML systems, GPU kernels, and fixes for the tools they depend on.

## Start with Loggetta

```bash
pip install loggetta
loggetta inspect
loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
```

**One install. Check the machine, plan a supported training run, and keep the results.**
Training on your own data, with reusable adapters, is [on main](https://github.com/pjordanandrsn/loggetta/blob/main/docs/TRAINING.md) and ships in 0.3.0.

| Project | What it does |
| :--- | :--- |
| [**Loggetta**](https://github.com/pjordanandrsn/loggetta) | Plans MoE training for your hardware and records what ran. |
| [**experts4bit-qlora**](https://github.com/pjordanandrsn/experts4bit-qlora) | Fine-tunes and serves large MoEs using GPU memory, RAM, and SSD storage. |
| [**grouped-nf4-gemm**](https://github.com/pjordanandrsn/grouped-nf4-gemm) | Runs expert math directly on packed 4-bit weights. |

## Results worth a look

- **2.352× training speed vs Unsloth** on Qwen3-30B-A3B / RTX 5090: 3.494 vs 8.218 s/step, matched work and software, comparable held-out loss. Unsloth used 3.22 GB less peak VRAM. [Benchmark](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-samestack-box4.md); [second host: 2.468×](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-samestack-host2.md).
- **120B QLoRA experiment at 9.82 GB peak VRAM**, using host memory. All 144 frozen expert tensors stayed byte-identical after training. [Native-MXFP4 experiment](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/mxfp4/RESULTS-mxfp4-train.md).
- **Long 4,096-token rows:** 1.453× vs Unsloth on an RTX 5090, with higher peak VRAM (defaults as of 2026-10-06). [Latest result and limits](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-packed4k-defaults.md).

I publish the hardware, code, controls, and failed tests alongside the wins. [Evidence and current status](https://cerinamroth.com/research/)

## Fixes merged upstream

| Project | Contribution |
| :--- | :--- |
| **OpenVINO** | Intel GPU compilation fixes and regression coverage: [#35661](https://github.com/openvinotoolkit/openvino/pull/35661), [#35712](https://github.com/openvinotoolkit/openvino/pull/35712), [#36017](https://github.com/openvinotoolkit/openvino/pull/36017). |
| **bitsandbytes / Axolotl** | Stop checkpointed training from retaining dequantized parameters in the cache: [#1999](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1999), [#3915](https://github.com/axolotl-ai-cloud/axolotl/pull/3915). |
| **Transformers / Unsloth Zoo** | Fix end-of-sequence handling and MoE weight preprocessing: [#47931](https://github.com/huggingface/transformers/pull/47931), [#1082](https://github.com/unslothai/unsloth-zoo/pull/1082). |
| **zkbugs** | Add the public ZisK signed-remainder soundness bug record: [#97](https://github.com/zksecurity/zkbugs/pull/97). |

Also: [JWST image research](https://cerinamroth.com/jwst/lrds/), [security disclosures](https://cerinamroth.com/disclosures/), and [technical and policy consulting](https://jordananderson.work).

<details>
<summary><strong>Contact and public GPG key</strong></summary>

[Email](mailto:jordan@cerinamroth.com) · [Download public key](https://jordananderson.work/.well-known/jordananderson-pubkey.asc) · [Verify on my website](https://jordananderson.work/#public-key)

Ed25519 · expires 11 June 2028.

```text
DAEF C03B 9188 793C 80E5 20C8 68F0 6663 C7FB 250F
```

</details>
