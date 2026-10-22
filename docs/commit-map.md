# Commit map

On 2026-10-22 the history was rewritten to remove `Co-Authored-By` trailers from commit messages. Trees, authors, and dates are unchanged; only the hashes changed. Run records (`results/*/env.json` → `git_commit`) and older docs cite the old hashes; this table maps them to the current ones. A bundle of the old history was kept locally (`~/Small_Inference.pre-rewrite.bundle`).

| Old hash | New hash | Date | Subject |
|---|---|---|---|
| `8ec1175ea1d5a6b3241020e22a82b77a3edb0002` | `b6a1d641ac19d8b99daa942ebfd4362fb32c6ddc` | 2026-10-10 | Initial commit: project spec, study guide, implementation plan |
| `69deb00338a2e119d159d5f6a204dd2bbb14cf69` | `4db1597396a541b32efcffa7ef77213e81aea18a` | 2026-10-10 | Add pinned Python environment (uv, torch 2.14.1+cu130) |
| `9a093cec1c1a646b759cf0e180cb24c1f882ae35` | `2b00b924212f48ec0d112013412eabe1bbc96a75` | 2026-10-10 | Finish Step 0 and add M0.1 Engine protocol and fake engine |
| `85351d166dc0419f7252bdf4e9d6f66e1cd01fd2` | `1713c2956e774f672cd6239f784f886ccaccf846` | 2026-10-11 | Add docstrings to every function and method |
| `561acbdb33a688034c2471133ccda83dfc9d60aa` | `a6d8ab1db881ed77eef3c50196012c970be2be2c` | 2026-10-11 | Add M0.2 open-loop runner and environment record |
| `f58fde29079a17b930dbcb8cb9ad6762cb8c0234` | `05ad1551a54831680a8dcf5707074717ac99cd7e` | 2026-10-11 | Add M0.3 metrics: raw run records to summary.json |
| `3f66a62a6efc077634d3662dc406f7e557a0e6c0` | `f1939fd6095b0d12fc63ad0d2d9dd4d09dc38130` | 2026-10-11 | Add M0.4 harness self-test against the fake engine |
| `deb053add180d2b338bb6fddeff9fd97e08cd1cb` | `3668a314de5fa6831c6798b1db7e562b3e83f51a` | 2026-10-11 | Add M0.5 ShareGPT trace builder |
| `3431b49dd08236267e99c76011730839771d5481` | `47b04ada5453ee41bedd7a8501641ed5aa2d97f5` | 2026-10-11 | Add ShareGPT burst trace (n = 100, seed 0) |
| `f9366adb582449ca0da0943bacd442d5d27e02cd` | `42bff420b8610e2c1565c15c3625c11f3848dede` | 2026-10-11 | Add M0.6 roofline measurement script |
| `1c320aa91e32fd65469f5880879787b4b81e8847` | `f54bd6520a1e237a3864c280a614a8ecfae0f841` | 2026-10-11 | Record M0.6 roofline results (3 runs) and fix the git-dirty check |
| `3fe4be110afe7500dfa0ab096965470ab67f883c` | `0131f9e6d99d883637dfd7a8fa5baa1d74615c64` | 2026-10-11 | Add M0.7 HF baseline engine and campaign driver; record roofline facts |
| `f39d9b72443a433e7b64f5be8019bf98c98e9428` | `d50b87d341477f59dd33c17e5fee0c698f37d6d4` | 2026-10-11 | Add HF baseline campaign results (Phase 0, M0.7) |
| `b914f250dddb93fca8a13e2b92b879fbb1f2e53d` | `dd636fab37aa5fcbc563e47594773ba68b00f25c` | 2026-10-11 | Add Phase 0 analysis script and figures for the HF baseline |
| `3e4a1a66bbf99617cd43384862de211aaff51245` | `4f65b60f3ff8cf1309614d21d96adfdccee4ce11` | 2026-10-11 | Record GitHub remote and the push workflow in the plan |
| `65bf2fbc90ad2ea40939326dd08a9c8b536944bb` | `f94ec88c3db3fa2e6cb2359238925739cb27e6ae` | 2026-10-19 | Add run telemetry, shuffled campaign order, temperature cooldown, decode profiler, and CPU probe |
| `0203ab87bd3bfbff5a0db3cde2554deecd72456c` | `6a6f684fe573a0b88b5afb263d4369f4063996e2` | 2026-10-19 | Phase 1: Qwen2 forward pass, contiguous KV cache, greedy generation, batch-1 engine |
| `0d14ba7dc638b882ad39546677a9fadab39ddfa6` | `8685493b5b090df157e233b9c18e08c5b3a546bd` | 2026-10-19 | Add a Phase 1 decode profile (smoke test in Balanced mode; kernel counts only) |
| `f89ec59f337e7c91862bb8829dca63fa43bfeda0` | `4653bc59012c35e72af303ea19c24dcd865ff78e` | 2026-10-19 | Add the deferred-runs queue and a multi-engine campaign driver |
| `45916d1356d931af6e482aeb9a99a6e018d98af0` | `13cff00b03a8a2358f64ace65f87310e4e60579d` | 2026-10-19 | Phase 2: block manager, paged KV cache, packed metadata, gather attention |
| `c4fb2e16304d16b35b2d9c81c23707d2660943fc` | `bb46f598b442551a7733e2eb3264e8ca25bda493` | 2026-10-19 | Phase 3: Triton paged-attention decode kernel with split-K |
| `f7a9cfc8324e1b98f82d619f7c9b2d77d77211d0` | `a7c5404c005036c8953768f011ca4551b1e43ed5` | 2026-10-19 | Phase 4: continuous batching with chunked prefill and recompute preemption |
| `2af4ae54ee452272211031991e81b7950b3b5203` | `90b84c0109d2a6307cb991ebe71fcc63efeebfc5` | 2026-10-19 | Add a Phase 4 batch-1 decode profile (smoke test in Balanced mode; kernel counts only) |
| `2999829aa9e40622b8c00741e7310e4dba94532f` | `662c58d3579f7c622d8bf1e27c8a0070a135150e` | 2026-10-19 | Phase 5: INT8 KV caches with dequantization fused into the decode kernel |
| `367c77b775cadbf221b636224ddce70b91538d64` | `7a1c03cae2db5ffa8b67da15a5c2bf6b59063387` | 2026-10-19 | Add Phase 5 quality results on Qwen2.5-0.5B (20 documents, 64-token chunks) |
| `72076d9edc7377d1b2508fa59e99cfee21c02588` | `da5268dcf080b4fa06370f2d675dab62c5fab9b4` | 2026-10-19 | Phase 6 tooling: vLLM adapter, one-command figure rebuild, deferred vLLM runs |
| `e8f2f9c6d5abbca21c8f309b4314234659f7bccf` | `1d7c0d53cc4b7fbc601bfe778db498d69d4f66a2` | 2026-10-19 | Record idle-laptop profiles and probes: bimodal TPOT is GPU DVFS |
| `1bff8c977d58390abf91457abb550587e6b349e6` | `6c89fea33e9ab1058c3e89b75a5b6f67a30d2d1c` | 2026-10-19 | Record locked-clock probes: decode variance is launch enqueue through WSL |
| `14dfb72b87ba59f03dc5a76ee1fccc6ff8820a95` | `76bd5c480b9564cf5a23bcc71c821553224fbdb7` | 2026-10-19 | paged_gather: measure GPU kernel time and kernel count, not just launch-bound wall time |
| `2437f9f4e0feb28626e022de4c12d0f6b8a8a9a4` | `029648135d19cdcef76138133dbcceef2784f1b3` | 2026-10-19 | Phase 3 measured: 99% of copy bandwidth at batch 32; split-K policy from data |
| `bc8a28738e808231bc215be5f6f8f7a34546ffe4` | `1ffcce3cd7b237bfa53fcc25d7288164460011e8` | 2026-10-19 | Phase 5 measured on 0.5B and 1.5B: per-channel K quality win, INT8 kernel speed loss |
| `cfde332dc75be3d4b875c3fd5cefe03fcb1f74ec` | `fd0bf06d95c6c19716735a19f638818a89099da3` | 2026-10-19 | Fix the INT8 decode kernel and the overload check; add --n and timing reps |
| `24c1e4636f4b013596fb0a9c5c8a1dd5ca22d46f` | `3ae29f83a0a44b34843441386bc83c3470299410` | 2026-10-19 | Phase 5 table on 0.5B and 1.5B with the fixed kernel (timing median of 3) |
| `20a5a4edadc6dd47436118e61e0374bdfdb49e92` | `8b280719f13ff10df4cda8f9e73f76a087f44908` | 2026-10-19 | Phase 4 capacity at n = 1000: 8.65 req/s; overload only at 1.2x |
| `91480599f43c0cd5f6f453f001215ab2775c3ddd` | `56ccce13d642dd2aaa1f2e111da46bd6cfe7b09e` | 2026-10-19 | Phase 4 measured: chunked prefill keeps up at 1.0x; rate-curve figure |
| `772861905d03453694710422dae1d7ca01969c13` | `4294d06fdbe689bed36587f6d49c66bbd95b9524` | 2026-10-19 | Add --max-seqs to campaigns so the KV pool, not the sequence count, can bind |
| `5de75f04b3803274de77ed81c9066610b051af85` | `cd5177ffb2782401de160837b5ea6dfd2efa2d01` | 2026-10-19 | Record engine memory-pressure stats per run; add per-process repetition driver |
| `5f1e06988bea6c639b8b418c2b5b07628d9e8b39` | `fee8f8ffb4020ea8f7e2a90e8036a3147a1731e0` | 2026-10-20 | Fused INT8 quantize-and-store kernels; record the first 1.5B serving run |
| `c364daeeb36aa99d2cde53bbeb50711f157f61f8` | `b1cd4ad60ffb51e23958e5b20a85833daec5e31c` | 2026-10-20 | Record the 11b rerun's completed repetition and the Windows Update restart that interrupted it |
| `9ff87cea0d810cbf4e4ff190dbab21242db80166` | `be52e8bb82bcf5dc1a4d5338253616100b537155` | 2026-10-21 | Log the re-lock and the changed power limit; rerun all 11b repetitions under one limit |
| `6c14e142604f0753c1432a41118a90003c549779` | `a9938d2e37c321f1bbcb8a94252054734ac75851` | 2026-10-21 | Phase 5 complete: 1.5B serving under memory pressure, figures and tables |
| `763959a751c02843f713c28e243ffb1ccd7ddd2f` | `fe8ec5c20f2e9c058c1ec7fa0dcef3596a77b088` | 2026-10-21 | Log the third clock re-lock |
| `f2b8483204f7131f3d95d2af3beb9eb10ae3effc` | `46410c93c38b07fca11f6254bf39f86a6ee158ba` | 2026-10-21 | vLLM installed and tested; per-mode memory matching and per-engine interpreters |
| `959f6ce5180a7a781a1a7f29cff595e05cabf776` | `4e964b57ac8db5219cb505fc08066ed3d7925d9b` | 2026-10-22 | process_reps: per-child timeout (kills the process group) and resume |
| `73833f1855bea23480db5097227e9c31e022db81` | `d2b3c6c6b044cb0e4b8b364136ff01d73bde58fe` | 2026-10-22 | Queue on hold after item 15, per the user |
| `41895f5ee62249d568115aac25d248d1b9e97a32` | `3f65150041852feb2e391c802ae38627c7f321e5` | 2026-10-22 | Stop all testing per the user; record what of item 15 is usable |
| `84931aa0ff464ef38d50a7cef25724811583dd37` | `4ef37ec584893a9e115887ce4dc87ae10ddd4349` | 2026-10-22 | Add README: what's built, correctness, measured results, limitations, and remaining work |

Earlier rewrite (2026-10-19, three then-unpushed commits): one run record cites a hash from before it.

| Hash cited | Same commit after the 2026-10-19 rewrite | Current hash |
|---|---|---|
| `3ae33fb62c3c8ca6f1c23e93005565e511170c56` | `0203ab8` | `6a6f684fe573a0b88b5afb263d4369f4063996e2` |
