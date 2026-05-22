# Jordan Anderson

I find the gap between what an institution's contract claims and what its implementation actually does, and write the patch.

Recent technical work: OpenVINO Intel GPU plugin — `__local`-pointer kernel compilation fixes and a shared-header API extension across LoRA, MoE, and fully-connected kernels, plus a regression test so the bug class can't silently return.

- [PR #35661](https://github.com/openvinotoolkit/openvino/pull/35661) — merged
- [PR #35712](https://github.com/openvinotoolkit/openvino/pull/35712) — merged
- [PR #36017](https://github.com/openvinotoolkit/openvino/pull/36017) — regression test, approved (awaiting merge)

Built [ov-impact-bench](https://github.com/pjordanandrsn/ov-impact-bench) to measure the GPU-vs-CPU-fallback impact the fix unlocks — latency, energy, throughput — validated on real Intel silicon.

More at [jordananderson.work](https://jordananderson.work).
